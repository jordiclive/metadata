from datasets import load_dataset
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"LOG.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.FATAL,
)

def list_files(directory_path):
    file_list = []
    for entry in os.scandir(directory_path):
        if entry.is_file():
            file_list.append(entry.name)

    return file_list

directory_path = '/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec'  # Replace this with your desired directory path
file_names = list_files(directory_path)
def log_print(message):
    logging.fatal(message)
    print(message)

for k in file_names:
    try:
        datasets = load_dataset(path='/fsx/home-jordiclive/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec', data_files=[k])
    except:
        log_print('ERROR: with file {}'.format(str(k)))