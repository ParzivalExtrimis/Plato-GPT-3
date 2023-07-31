from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers
import os
import argparse
import mlflow
from azureml.core import Run
from utils.web_handler import Web_handler
from tokenizers.pre_tokenizers import Whitespace


def main(args):
    use_mlflow = True
    # Start Logging
    run = Run.get_context()
    run_id = run.id

    if run_id.startswith('OfflineRun'):
        use_mlflow = False
    else:
        mlflow.start_run(run_id=run_id)



    dataset = load_dataset("bookcorpus", split='train')

    #get web handle to upload the trained tokenizer
    w_handle = Web_handler(config=args.config)
    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = Whitespace()

    trainer = trainers.BpeTrainer(show_progress=True, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    bpe_tokenizer.train_from_iterator(dataset['text'], trainer=trainer)

    if use_mlflow:
        mlflow.log_text(bpe_tokenizer.to_str(pretty=True), 'tokenizer.json')
        mlflow.log_metric("Vocab Size: ", bpe_tokenizer.get_vocab_size(with_added_tokens=True))

    bpe_tokenizer.save(path=os.path.join("trained_tokenizer.json"))
    w_handle.upload_to_datastore(
        filepath = os.path.join("trained_tokenizer.json"),
        name = 'BPE-Tokenizer-BookCorpus',
        description = 'BPE-Tokenizer for the BookCorpus dataset, trained.',
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Location of config file or mount.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)