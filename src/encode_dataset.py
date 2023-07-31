from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datasets import load_dataset
from tokenizers import Tokenizer
import json
import os
import argparse
import mlflow
from azureml.core import Run
from utils.web_handler import Web_handler


def main(args):
    use_mlflow = True
    # Start Logging
    run = Run.get_context()
    run_id = run.id

    if run_id.startswith('OfflineRun'):
        use_mlflow = False
    else:
        mlflow.start_run(run_id=run_id)


    dataset_name = "bookcorpus"
    dataset = load_dataset(dataset_name, split='train')

    #get web handle to upload the trained tokenizer
    w_handle = Web_handler(config=args.config)

    os.makedirs('tokenizer', exist_ok=True)
    #download the trained tokenizer
    w_handle.download_from_datastore(
        name = 'BPE-Tokenizer-BookCorpus',
        save_dir='tokenizer',
    )

    bpe_tokenizer = Tokenizer.from_file(os.path.join('tokenizer','trained_tokenizer.json'))
    encodeds = bpe_tokenizer.encode_batch(dataset['text'])

    if use_mlflow:
        mlflow.log_text(str(encodeds[:10]), 'generated_encodings_sample.json')


    # save (token IDs) from encodings and upload
    encoded_ids = [encoded.ids for encoded in encodeds]
    with open("encoded_ids.json", "w") as f:
        json.dump(encoded_ids, f)

    w_handle.upload_to_datastore(
        filepath = "encoded_ids.json",
        name = f"{dataset_name}-encoded_ids",
        description = f'{dataset_name} dataset encoded in BPE.',
    )
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Location of config file or mount.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)