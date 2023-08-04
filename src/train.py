import datetime
import os
import pytz
import time
import json
import math
import subprocess
import argparse
import mlflow
import mlflow.pytorch
import torch

from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer
from azureml.core import Run
from typing import List, Tuple

from model.model import GPTLanguageModel, Config
from utils.web_handler import Web_handler
from utils.dataloader import BookCorpusDataset


def format_time(secs: float) -> str:
    return time.strftime('%H:%M:%S', time.gmtime(secs))
    
def main(args):

    torch.manual_seed(1337)
    use_mlflow = True
    # Start Logging
    run = Run.get_context()
    run_id = run.id


    if run_id.startswith('OfflineRun'):
        use_mlflow = False
    else:
        mlflow.start_run(run_id=run_id)

    if use_mlflow:
        mlflow.autolog()

    #state vars
    epoch = 0
    init_epoch = 0
    output_line_count = 10
    save_point_datastore_name = 'checkpoint-BPE-DDP'

    #ddp vars
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    multinode_available = world_size > 1
    self_is_main_node = world_rank == 0
    device = None # uninitialised until DDP init

    #dataloading
    dataload_workers = args.dataload_workers
    prefetch_factor = args.prefetch_factor

    # hyperparameters
    start_fresh = args.start_fresh
    always_override_checkpoint = args.always_override_checkpoint

    split_ratio = args.split_ratio
    batch_size = args.batch_size 
    block_size = args.block_size
    max_epoch = args.max_epoch
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    eval_iters = args.eval_iters
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    bias = args.bias

    use_decay = args.use_decay
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2
    learning_rate = args.learning_rate
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters
    min_lr = args.min_lr

    config_f = args.config
    dataset_name = args.dataset_name
    # ------------

    w_handle = Web_handler(config_f)

    ### ddp configs

    # Use CUDA if it is available
    if torch.cuda.is_available():
        print(f"Setting up torch.device for CUDA for local gpu:{local_rank}")
        device = str(torch.device(local_rank))
    else:
        print(f"Setting up torch.device for cpu")
        device = str(torch.device("cpu"))

    if multinode_available:
        print(
            f"Running in multinode with backend = 'nccl' local_rank={local_rank} rank={world_rank} size={world_size}"
        )
        torch.distributed.init_process_group(
            "nccl",
            rank= world_rank,
            world_size= world_size,
        )
    else:
        print(f"Not running in multinode.")

    if dataload_workers < 0:
        dataload_workers = os.cpu_count()

    # to log the dataset being used
    dataset = {
        'name' : dataset_name,
    }

    # log params
    params = {
            'start_fresh' : start_fresh,
            'always_override_checkpoint' : always_override_checkpoint,
            'split_ratio' : split_ratio,
            'dataload_workers': dataload_workers,
            'batch_size' : batch_size, 
            'block_size' : block_size,
            'max_epcoh' : max_epoch,
            'eval_interval' : eval_interval,
            'save_interval' : save_interval,
            'learning_rate' : learning_rate,
            'eval_iters' : eval_iters,
            'n_embd' : n_embd,
            'n_head' : n_head,
            'n_layer' : n_layer,
            'dropout' : dropout,
            'bias' : bias, 
            'use_decay' : use_decay,
            'weight_decay' : weight_decay,
            'beta1' : beta1,
            'beta2' : beta2,
            'learning_rate' : learning_rate,
            'warmup_iters' : warmup_iters,
            'lr_decay_iters' : lr_decay_iters,
            'min_lr' : min_lr,
            'world_size' : world_size,
            'world_rank' : world_rank,
            'local_world_size' : local_world_size,
            'local_rank' : local_rank,
            'multinode_available' : multinode_available,
            'device' : device,
    }

    # only log if running on main node to prevent duplication
    if use_mlflow and self_is_main_node:
        mlflow.log_params(params)

    #ensure download dirs exist
    ids_dir = os.path.join('encoded-ids')
    tokenizer_dir =  os.path.join('tokenizer')
    os.makedirs(ids_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    # download pre-encoded dataset encodings
    w_handle.download_from_datastore(
        name = f'{dataset_name}-encoded_ids',
        save_dir = ids_dir
    )

    # download pre-trained tokenizer
    w_handle.download_from_datastore(
        name = 'BPE-Tokenizer-BookCorpus',
        save_dir = tokenizer_dir
    )
    
    #get tokenizer
    tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, 'trained_tokenizer.json'))

    # load ids into a list from json
    with open(os.path.join(ids_dir, 'encoded_ids.json'), 'r') as f:
        ids = json.load(f)
    print('Tokens loaded: ', ids[:5])

    # get tensors from list
    ids = [n for line in ids for n in line]
    print('Tokens flattened: ', ids[:25])
    token_count = len(ids)
    print('Total token count: ', token_count)

    def get_split(ids: List[int], ratio: int) -> Tuple[List[int], List[int]]:
        n = int(ratio * len(ids)) # first 90% ( split% ) will be train, rest val
        train_data = ids[:n]
        val_data = ids[n:]

        return train_data, val_data

    train_data, val_data = get_split(ids, ratio=split_ratio)

    print('sample -- train: ', train_data[:10])
    print('sample -- val:   ', val_data[:10])

    train_dataset = BookCorpusDataset(train_data, block_size=10)
    val_dataset = BookCorpusDataset(val_data, block_size=10)

    #ddp - sampling training dataset
    training_dataset_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=world_rank
    )
    # prefetch
    optional_data_loading_kwargs = {}
    if dataload_workers > 0:
            optional_data_loading_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers = dataload_workers,
            pin_memory=True,
            sampler = training_dataset_sampler,
            **optional_data_loading_kwargs,
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers = dataload_workers,
            pin_memory=True,
    )
    
    # with dataloader
    @torch.no_grad()
    def estimate_loss(split: str):
        model.eval()
        loader = train_loader if split == 'train' else val_loader
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(iter(loader))
            _, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out = losses.mean()
        model.train()
        return out
        
    
    # make checkpoints dir for instance; at fresh use for saving checkpoints;
    #                                    at resume use for downloading and loading checkpoints as well as saving
    checkpoints_dir = os.path.join('checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # set parameter options to be set for  each train instance
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        bias=bias, vocab_size=None, dropout=dropout)

    #starting from scratch
    if start_fresh:
        print('Starting model training from scratch...')
        model_args['vocab_size'] = tokenizer.get_vocab_size(with_added_tokens=True)

        conf = Config(**model_args)
        model = GPTLanguageModel(conf)
        m = model.to(device)
        best_val_loss = 1e9

    else:
        # download saved instances from DataStore
        w_handle.download_from_datastore(
            name = save_point_datastore_name,
            save_dir = checkpoints_dir,
        )
        checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            if model_args[k] != checkpoint_model_args[k]:
                print(f'New parameters fount. Does not match checkpoint. Run mode set to [Start_fresh = {start_fresh}]. Overwriting parameters - {k}, Old value: {model_args[k]} - New value: {checkpoint_model_args[k]}')
                if use_mlflow:
                    mlflow.log_param(f'new-{k}', checkpoint_model_args[k])
            model_args[k] = checkpoint_model_args[k]

        # create the model
        conf = Config(**model_args)
        model = GPTLanguageModel(conf)
        state_dict = checkpoint['model']
        
        # anomalous prefix in state_dict, removed.
        unwanted_prefix = '_orig_mod.'
        for k,_ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        m = model.to(device=device)

        epoch = checkpoint['epoch']
        init_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f'Resuming model training from checkpoint [saved at: {checkpoint["timestamp"]}]')

    #ddp - wrap model with ddp
    m = DDP(m)

    if not always_override_checkpoint and start_fresh:
        # download saved instances from DataStore
        try:
            w_handle.download_from_datastore(
                name = save_point_datastore_name,
                save_dir = checkpoints_dir,
            )
            checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)
            best_val_loss = checkpoint['best_val_loss']
        except Exception as e:
            print(e, '\n\n','Could not find checkpoint at specified location at the Datastore. Continuing with fresh best loss...')
        finally:
            pass

    #decayed learning rate
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    #=========================================================================================================

    # create a PyTorch optimizer
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    if not start_fresh:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    start = time.time()

    total_epoch = init_epoch + max_epoch
    if use_mlflow and self_is_main_node:
        mlflow.log_param('total_epoch', total_epoch)

    last_train_loss = 1e9
    last_val_loss = 1e9

    while True:

        curr_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        
        lr = get_lr(epoch) if use_decay else learning_rate

        if use_mlflow and self_is_main_node:
            mlflow.log_metric('learning rate', lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # every once in a while evaluate the loss on train and val sets
        if epoch % eval_interval == 0:
            last_train_loss = estimate_loss('train')
            last_val_loss = estimate_loss('val')
            if use_mlflow and self_is_main_node:
                mlflow.log_metric('Training Loss', last_train_loss)
                mlflow.log_metric('Validation Loss', last_val_loss)

            print(f"GPU[{local_rank}] - Step {epoch}: Train loss {last_train_loss:.4f}, Val loss {last_val_loss:.4f}")

        if epoch % (eval_interval*4) == 0 and use_mlflow and device != 'cpu':
             #track gpu stats
            gpu_stats = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            mlflow.log_text(gpu_stats, os.path.join('gpu_stats', f'GPU-{local_rank}', f'{curr_time}.txt'))

        if epoch % save_interval == 0 and epoch > 0 and self_is_main_node:
            # save checkpoints 
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss # set new record, save state 

                # ddp - check for DDP wrapping
                if isinstance(m, DDP):
                    model = m.module.to('cpu')
                else:
                    model = m.to('cpu')

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'epoch': epoch,
                    'model_parameters': model.get_num_params(),
                    'best_val_loss': best_val_loss,
                    'timestamp' : curr_time,
                }

                # mlflow log model
                if use_mlflow:
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="saved_model",
                        registered_model_name=f"{model.get_num_params()/1e6:.2f}M-param-model",  # also register it if name is provided
                    )

                print(f"saving checkpoint to {os.path.abspath(checkpoints_dir)}")
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'checkpoint.pt'))

                w_handle.upload_to_datastore(
                    filepath = os.path.join(checkpoints_dir, 'checkpoint.pt'),
                    name = save_point_datastore_name,
                    description = 'Checkpoint for torch.save() last commit -- for the current tokenizer_type.',
                )
                checkpoint = None # free up memory
        
        # sample a batch of data
        xb, yb = next(iter(train_loader))

        # evaluate the loss
        _, loss = model(xb.to(device), yb.to(device))
        optimizer.zero_grad(set_to_none=True)
        #backward pass
        loss.backward()
        # step
        optimizer.step()

        epoch += 1
        if(epoch > total_epoch):
            break

        # testing only pause, remove at training
        break

    end = time.time()

    #==========================================================================================================
    if self_is_main_node:

        #ddp - check for DDP wrappings
        if isinstance(m, DDP):
            model = m.module

        # ouput logs directory
        logs_dir = os.path.join('outputs', str(run.display_name))
        os.makedirs(logs_dir, exist_ok=True)
        out_text_path = os.path.join(logs_dir, 'out.txt')

        # generate from the model; and logging
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print('\n---------------------------------------------\n')
        for ln in range(output_line_count):
            print(tokenizer.decode(model.generate(context, max_new_tokens=20)[0].tolist()))
        print('\n---------------------------------------------\n')


        with open(out_text_path, 'w') as out_f:
            out_f.write(tokenizer.decode(model.generate(context, max_new_tokens=5000)[0].tolist()))


        if use_mlflow:
            mlflow.log_artifact(
                local_path=out_text_path,
                artifact_path='generated-text'
            )


        elapse_interval_sec = end - start
        elapsed_time = format_time(elapse_interval_sec)

        snapshot = {
            'num_model_params' : model.get_num_params(),
            'vocab_size' : model_args['vocab_size'],
            'token_count' : token_count,
            'params' : params,
            'dataset' : dataset,
            'training_time': elapsed_time,
            'last_train_loss' : last_train_loss.item(),
            'last_val_loss' : last_val_loss.item(),
            'best_val_loss' : best_val_loss,
        }

        
        with open(os.path.join(logs_dir,'run_meta.json'), 'w') as fp:
            json.dump(snapshot, fp, indent=4)

        if use_mlflow:
            mlflow.log_artifact(os.path.join(logs_dir,'run_meta.json'))

    if use_mlflow:    
        mlflow.end_run()

    run.complete()

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='config.json', help='Location of Config file or mount.')
    parser.add_argument('--dataset_name', type=str, default='bookcorpus', required=False, help='Name of the dataset used to reference Data Asset Store.')
    parser.add_argument('--start_fresh', type=bool, default=True, required=False, help='Flag indicates whether to use checkpoints to load at training.')
    parser.add_argument('--always_override_checkpoint', type=bool, default=False, required=False, help='Flag indicates whether to override checkpoints even when the current loss is higher than overall best at training.')

    parser.add_argument('--split_ratio', type=float, default=0.7, required=False, help='Ratio to split train and val data, in the form (train/val).')
    parser.add_argument('--dataload_workers', type=int, default=-1, required=False, help='Number of workers to use in parallel dataloading.')
    parser.add_argument('--prefetch_factor', type=int, default=2, required=False, help='Number of processes to use in-order to prefetch data.')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Number of parallel examples to be used per epoch.')
    parser.add_argument('--block_size', type=int, default=512, required=False, help='Context window size of the transformer.')
    parser.add_argument('--max_epoch', type=int, default=50000, required=False, help='Total number of iterations for training.')
    parser.add_argument('--eval_interval', type=int, default=1000, required=False, help='Iterations to wait until next loss evaluation.')
    parser.add_argument('--save_interval', type=int, default=5000, required=False, help='Iterations to wait until next checkpoint save.')
    parser.add_argument('--use-cuda', type=bool, default=True, required=False, help='Flag indicates whether to use CUDA at training.')
    parser.add_argument('--eval_iters', type=int, default=750, required=False, help='Number of samples to use in-order to smooth out loss over batches.')
    parser.add_argument('--n_embd', type=int, default=1536, required=False, help='Size of the embedding dimension.')
    parser.add_argument('--n_head', type=int, default=24, required=False, help='Number of attention heads.')
    parser.add_argument('--n_layer', type=int, default=16, required=False, help='Number of times to loop over tranformer layers.')
    parser.add_argument('--dropout', type=float, default=0.0, required=False, help='Dropout Ratio')
    parser.add_argument('--bias', type=bool, default=True, required=False, help='Flag indicates whether to use biases in Linear and LayerNorm layers.')

    # optimizer args
    parser.add_argument('--use_decay', type=bool, default=True, required=False, help='Flag indicated whether to use learning rate decay ( cosine decay ).')
    parser.add_argument('--learning_rate', type=float, default=6e-4, required=False, help='The magnitude at which the optimizer step changes the weights.')
    parser.add_argument('--weight_decay', type=float, default=1e-1, required=False, help='The magnitude at which the optimizer step changes the weights.')
    parser.add_argument('--beta1', type=float, default=0.9, required=False, help='Variable controls decay parameters.')
    parser.add_argument('--beta2', type=float, default=0.95, required=False, help='Variable controls decay parameters.')
    parser.add_argument('--warmup_iters', type=int, default=2, required=False, help='Initial iterations to run linear lr increment upto default lr.')
    parser.add_argument('--lr_decay_iters', type=int, default=45000, required=False, help='The amount of iterations upto which decay applies. Defaults to min_l after.')
    parser.add_argument('--min_lr', type=float, default=3e-6, required=False, help='The magnitude at which the optimizer step changes the weights.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)