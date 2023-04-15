from pathlib import Path

from datasets import load_dataset
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"LOG_only_good.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.FATAL,
)
import os
os.environ['TRANSFORMERS_CACHE'] = '/admin/home-jordiclive/transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/fsx/home-jordiclive/hf_datasets_cache'
def list_files(directory_path):
    file_list = []
    for entry in os.scandir(directory_path):
        if entry.is_file():
            file_list.append(entry.name)

    return file_list

directory_path = '/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec'  # Replace this with your desired directory path
file_paths = list_files(directory_path)
def log_print(message):
    logging.fatal(message)
    print(message)
#
# for k in file_names:
#     try:
#         datasets = load_dataset(path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec', data_files=[k])
#         log_print('np file {}'.format(str(k)))
#     except:
#         log_print('ERROR: with file {}'.format(str(k)))

from bsmetadata.experiments.datasetv2 import data_files_with_entities

local_dir = "/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec"
file_paths = list(Path(local_dir).glob("*.jsonl.gz"))

files_with_entities = [x for x in file_paths if x.name in data_files_with_entities]
files_without_entities = [x for x in file_paths if x.name not in data_files_with_entities]
print(f"{len(files_with_entities)} files with entities")
print(f"{len(files_without_entities)} ")

train_files = [x.name for x in files_with_entities if 'c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz' not in x.name]
# train_files = train_files[:2]

val_files = [x.name for x in files_with_entities if 'c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz' in x.name]
# train_files = train_files[:2]

errors = ['c4-en-html_cc-main-2019-18_pq00-177.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-214.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq01-000.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-117.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq00-137.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-120.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq00-131.jsonl.gz', '.gitattributes',
          'c4-en-html_cc-main-2019-18_pq00-159.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-157.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq00-231.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-123.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq00-017.jsonl.gz', 'c4-en-html_cc-main-2019-18_pq00-028.jsonl.gz',
          'c4-en-html_cc-main-2019-18_pq00-234.jsonl.gz']
a = []
for k in train_files:
    for l in errors:
        if l in k:
            a.append(k)

a = list(set(a))

train_files = [x for x in train_files if x not in a]

print(len(train_files))
print(train_files)
import time
x = time.time()
datasets = load_dataset(
    path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec',
    data_files=train_files)

val = load_dataset(path=local_dir,data_files=val_files)
datasets['validation'] = val['train']
log_print('Time to load dataset: {}'.format(time.time() - x))

import functools
import logging
from pathlib import Path

from datasets import config, load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from bsmetadata.experiments.datasetv2 import data_files_with_entities


logger = logging.getLogger(__name__)


def preprocess_no_metadata(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    remove_cols = dataset.column_names["train"] if "train" in dataset.column_names else dataset.column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=12,
        remove_columns=remove_cols,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
        batch_size=1,
    )

    block_size = 1024

    def group_texts(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            padding_len = block_size - total_length
            result = {
                "input_ids": [concatenated_examples["input_ids"] + [tokenizer.eos_token_id] * padding_len],
                "attention_mask": [concatenated_examples["attention_mask"] + [0] * padding_len],
            }
        return result

    result = tokenized_dataset.map(
        functools.partial(group_texts, block_size=block_size),
        batched=True,
        num_proc=12,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size=1,
    )
    return result
    # return tokenized_dataset

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("jordiclive/bs-modeling-metadata_all_2000_steps")
datasets = preprocess_no_metadata(datasets, tokenizer)

datasets.save_to_disk('/admin/home-jordiclive/whole_processed_dataset')






# for k in train_files:
#     try:
#         datasets = load_dataset(path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec', data_files=[k])
#         log_print('np file {}'.format(str(k)))
#     except:
#         log_print('ERROR: with file {}'.format(str(k)))