import huggingface_hub
for k in ['c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz']:
    huggingface_hub.hf_hub_download(repo_id="bs-modeling-metadata/c4-en-html-with-training_metadata_all",
                                    filename=k, repo_type='dataset',
                                    cache_dir='local-data')