from datasets import load_dataset
from tokenizers import Tokenizer, Encoding
import os
import argparse
import mlflow
import torch
from azureml.core import Run
from utils.web_handler import Web_handler
from model.model import GPTLanguageModel, Config


def main(args):
    use_mlflow = True
    # Start Logging
    run = Run.get_context()
    run_id = run.id

    if run_id.startswith('OfflineRun'):
        use_mlflow = False
    else:
        mlflow.start_run(run_id=run_id)

    checkpoints_dir = 'checkpoints'
    tokenizer_dir = 'tokenizer'
    out_text_path = 'out.txt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    w_handle = Web_handler(args.config)

    #tokenizer 
    w_handle.download_from_datastore(
        name = 'BPE-Tokenizer-BookCorpus',
        save_dir= 'tokenizer'
    )
    tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, 'trained_tokenizer.json'))
    enc: Encoding = tokenizer.encode(args.first_tokens)
    print(enc.ids, '\n', enc.tokens)

    # download checkpoints from DataAsset Store
    w_handle.download_from_datastore(
        name = 'Checkpoint-BPE',
        save_dir= checkpoints_dir
    )

    checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'), map_location=device)
    model_args = checkpoint['model_args']

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
    m.eval()

    # generate
    with torch.no_grad():
        ids = enc.ids
        if len(enc.ids) > model_args['block_size']:
            print(f'Entered prompt string is larger than the allowed context length [{model_args["block_size"]}]. Cropping overflow ...')
            ids = enc.ids[: model_args["block_size"]]

        context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        print('context: ', context.shape)
        with open(out_text_path, 'w') as out_f:
            token_ids = m.generate(context, max_new_tokens=8000)[0].tolist()
            out_f.write(tokenizer.decode(token_ids))

    mlflow.log_artifact(out_text_path, 'genrated_out.txt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Location of config file or mount.')
    parser.add_argument('--first_tokens', type=str, default='Once upon a time', help='String to start with.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)