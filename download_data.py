import huggingface_hub
# for k in ['c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz',  'c4-en-html_cc-main-2019-18_pq00-004.jsonl.gz',
# 'c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz',  'c4-en-html_cc-main-2019-18_pq00-005.jsonl.gz',
# 'c4-en-html_cc-main-2019-18_pq00-002.jsonl.gz' , 'c4-en-html_cc-main-2019-18_pq01-053.jsonl.gz',
# 'c4-en-html_cc-main-2019-18_pq00-003.jsonl.gz']:
#     huggingface_hub.hf_hub_download(repo_id="bs-modeling-metadata/c4-en-html-with-metadata",
#                                     filename=k, repo_type='dataset',
#                                     cache_dir='local-data')

import huggingface_hub

huggingface_hub.snapshot_download(repo_id='bs-modeling-metadata/c4-en-html-with-training_metadata_all',cache_dir='local-data')

