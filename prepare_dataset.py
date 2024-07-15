from huggingface_hub import hf_hub_download, snapshot_download


snapshot_download(
    "bs-modeling-metadata/c4-en-html-with-training_metadata_all",
    local_dir="/home/jordan/MRs/metadata/local_data",
    local_dir_use_symlinks=False,
    token="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",
    repo_type="dataset",
)
