import re
from huggingface_hub import HfApi
import glob
import os
from huggingface_hub import HfApi


def upload_folders_missing(user, repo_name):
    hf_api = HfApi()

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






if __name__ == '__main__':
    user = "bs-modeling-metadata"
    repo_name = "checkpoints_all_06_12_html_0.5_special_tokens"
    full_repo_name = f"{user}/{repo_name}"
    ckpt_dir = "/fsx/home-jordiclive/tmp/metadata-html-half"

    hf_api = HfApi()
    try:
        # Get the repository details
        repo_info = hf_api.repo_info(full_repo_name)
    except Exception:
        # If the repo doesn't exist, create it
        hf_api.create_repo(full_repo_name)
        repo_info = hf_api.repo_info(full_repo_name)

    import time
    while True:
        import os
        os.chdir(ckpt_dir)
        folders_to_upload = upload_folders_missing(user, repo_name)
        print(folders_to_upload)
    #     for model in folders_to_upload:
    #         hf_api.upload_folder(
    #
    #             folder_path=model,
    #
    #             path_in_repo=".",
    #
    #             repo_id=full_repo_name,
    #
    #             repo_type="model",
    #
    #         )
        time.sleep(3600)