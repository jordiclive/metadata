import re
from huggingface_hub import HfApi
import glob
import os
from huggingface_hub import HfApi


def upload_folders_missing(user, repo_name):
    hf_api = HfApi()

    subfolder = "main"

    repo_id = f"{user}/{repo_name}"

    # Get the repository details
    repo_info = hf_api.repo_info(repo_id)

    # Get the file list from the repository
    output = repr(repo_info)
    checkpoint_folders = re.findall(r'checkpoint-\d+step', output)

    # Convert the set to a list and remove duplicates
    unique_checkpoint_folders = list(set(checkpoint_folders))

    x = glob.glob('*')
    folders_to_upload = set(list(x)) - set(list(unique_checkpoint_folders))
    folders_to_upload = [i for i in folders_to_upload if ('checkpoint' in i)]

    folders_to_remove = set(list(unique_checkpoint_folders)).intersection(set(list(x)))
    return list(folders_to_upload), list(folders_to_remove)


def get_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def filter_smaller_folders(folders, max_size_gb=5):
    max_size_bytes = max_size_gb * (1024 ** 3)
    smaller_folders = []

    for folder in folders:
        if os.path.isdir(folder):
            folder_size = get_size(folder)
            if folder_size < max_size_bytes:
                smaller_folders.append(folder)

    return smaller_folders

# Usage example


def process_uploads(repo_name,ckpt_dir,user="bs-modeling-metadata"):
    os.chdir(ckpt_dir)
    upload_folders, remove_folders = upload_folders_missing(user, repo_name)
    x = filter_smaller_folders(upload_folders, 5)
    api = HfApi()
    for i in x:
        api.upload_folder(

            folder_path=i,

            path_in_repo=i,

            repo_id=f"{user}/{repo_name}",

            repo_type="model",

        )
    for i in remove_folders:
        os.system(f'rm -rf {i}')


repo_name = "checkpoints_all_04_23_global_only"
ckpt_dir = "/fsx/home-jordiclive/tmp/metadata-run-html-no-global"

user = "bs-modeling-metadata"
repo_name = "checkpoints_all_04_23_no_html"
ckpt_dir = "metadata-run-no-html"

user = "bs-modeling-metadata"
repo_name = "checkpoints_all_04_23_no_entity"
ckpt_dir = "metadata-run-no-entity_para"

if __name__ == '__main__':
    import time
    while True:
        process_uploads(repo_name="checkpoints_all_04_23_no_html", ckpt_dir="/fsx/home-jordiclive/tmp/metadata-run-no-html")
        process_uploads(repo_name="checkpoints_all_04_23_no_entity", ckpt_dir="/fsx/home-jordiclive/tmp/metadata-run-no-entity_para")
        process_uploads(repo_name="checkpoints_all_04_23_global_only", ckpt_dir="/fsx/home-jordiclive/tmp/metadata-run-html-no-global")
        time.sleep(3600)