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
os['TRANSFORMERS_CACHE'] = '/admin/home-jordiclive/transformers_cache'
os['HF_DATASETS_CACHE'] = '/fsx/home-jordiclive/hf_datasets_cache'
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

datasets = load_dataset(
    path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec',
    data_files=train_files)

# for k in train_files:
#     try:
#         datasets = load_dataset(path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec', data_files=[k])
#         log_print('np file {}'.format(str(k)))
#     except:
#         log_print('ERROR: with file {}'.format(str(k)))