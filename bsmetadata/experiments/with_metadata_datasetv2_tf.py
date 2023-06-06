import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from bsmetadata.experiments.datasetv2 import data_files_with_entities
from bsmetadata.metadata_utils import add_metadata_and_chunk_examples, random_sample_metadata_v2


logger = logging.getLogger(__name__)

tf.config.set_visible_devices([], "GPU")  # tell tensorflow not to use the GPU


def get_dataset(file_paths, num_gpus, gpu_id, data_config, tokenizer):
    data = tf.data.Dataset.from_tensor_slices([str(x.resolve()) for x in file_paths])

    if len(file_paths) >= num_gpus:
        data = data.shard(num_gpus, gpu_id)
    # add shuffle buffer size equal to the number of files, so each epoch is a different shuffle
    buffer_size = len(file_paths) if len(file_paths) < num_gpus else len(file_paths) // num_gpus
    data = data.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=True)
    data = data.interleave(
        lambda x: tf.data.TextLineDataset(x, compression_type="GZIP"),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4,
        block_length=16,
    )
    # Reads 4 files concurrently, and interleave blocks of 16 lines

    if len(file_paths) < num_gpus:
        data = data.shard(num_gpus, gpu_id)

    def from_json_string(t):
        string = t.numpy().decode("utf-8")
        x = json.loads(string)
        if not x["text"]:
            return np.array([])
        example = x
        # add columns if they don't exist
        for col in ["metadata_entity", "metadata_entity_paragraph"]:
            if col not in example:
                example[col] = []

        # drop columns not specified in config
        keep_metadata_columns = [f"metadata_{key}" for key in data_config.metadata_config.metadata_column_list]
        remove_columns = [
            key for key in example.keys() if key.startswith("metadata_") and key not in keep_metadata_columns
        ]
        example = {k: v for k, v in example.items() if k not in remove_columns}

        examples = {k: [v] for k, v in example.items()}
        metadata_type_sample_weights = data_config.metadata_config.random_sample_metadata_weights
        examples = random_sample_metadata_v2(
            examples,
            metadata_type_sample_weights=metadata_type_sample_weights,
            html_overall_sample_rate=data_config.metadata_config.html_overall_sample_rate,
        )
        # example = {k: v[0] for k, v in examples.items()}

        result = add_metadata_and_chunk_examples(examples, tokenizer, data_config.metadata_config)

        # TODO: consider removing examples too short

        return np.stack(
            [
                result["input_ids"],
                result["attention_mask"],
                result["metadata_mask"],
            ],
            axis=1,
        )  # batch size, 3, seq len

    data = data.map(lambda x: tf.py_function(from_json_string, [x], tf.int32), num_parallel_calls=tf.data.AUTOTUNE)

    def filter_empty(t):
        return t.shape[0] > 0

    data = data.filter(lambda x: tf.py_function(filter_empty, [x], tf.bool))
    data = data.unbatch()
    #    data = data.batch(batch_size)
    #    data.prefetch(tf.data.AUTOTUNE)
    # don't batch and prefetch because we want to do it later, after mixing 2 datasets
    return data


def get_dataloader(*, tokenizer, args, num_gpus, gpu_id, train=True):
    """returns a tensorflow dataloader"""
    data_config = args
    local_dir = Path(data_config.dataset_name)
    # assert local_dir exists
    assert (
        local_dir.exists()
    ), f"data_config.dataset_name {local_dir} does not exist, it should be a local path for this dataloader to work"

    file_paths = list(Path(local_dir).glob(data_config.train_file))
    assert len(file_paths) > 0, f"no files found for {data_config.train_file}"

    files_with_entities = [x for x in file_paths if x.name in data_files_with_entities]
    files_without_entities = [x for x in file_paths if x.name not in data_files_with_entities]
    print(f"{len(files_with_entities)} files with entities")
    print(f"{len(files_without_entities)} files without entities")

    if train:
        files_with_entities = [
            x for x in files_with_entities if "c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz" not in x.name
        ]
    else:
        files_with_entities = [
            x for x in files_with_entities if "c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz" in x.name
        ]

    data_with_entities = get_dataset(files_with_entities, num_gpus, gpu_id, data_config, tokenizer)

    data = tf.data.Dataset.sample_from_datasets(
        [data_with_entities],
        weights=[float(len(files_with_entities))],
        seed=42,
    )
    data = data.shuffle(1000, reshuffle_each_iteration=True)
    data = data.batch(data_config.per_device_train_batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)

    def to_dict(t):
        batch = {
            "input_ids": t[:, 0, :],
            "labels": t[:, 0, :],
            "attention_mask": t[:, 1, :],
            "metadata_mask": t[:, 2, :],
        }
        batch = {k: torch.tensor(v.numpy(), dtype=int) for k, v in batch.items()}
        return batch

    return data, to_dict


def get_dummy_dataloader(batch_size):
    """returns a dummy pytorch dataloader, for accelerate to prepare"""
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(batch_size, 3, 512, dtype=torch.int32)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

if __name__ == '__main__':
    import yaml

    data_config_yaml = """
    streaming: True
    validation_size_max: 1024
    use_full_evaluation_for_val: true
    metadata_config:
      random_sample_metadata: true
      random_sample_metadata_calculate_size: 16384
      random_sample_metadata_weights:
        html: 1.0
        timestamp: 11.335125448028673
        website_desc: 10.532889258950874
        title: 1.0657717366883845
        generation_datasource: 1.0
        entity_paragraph: 1.028817740667444
        generation_length_text: 1.0
      metadata_list:
      - html
      - timestamp
      - website_description
      - title
      - url
      - datasource
      - length
      - entity_paragraph
      - generation_length_text
      metadata_column_list:
      - html
      - timestamp
      - website_desc
      - title
      - generation_datasource
      - generation_length_text
      - entity_paragraph
      local_metadata_special_tokens:
        entity_paragraph: "entity"
      metadata_sep: ' | '
      metadata_key_value_sep: ': '
      metadata_probability: 0.5
      treat_local_metadata_as_regular_text: true
      add_local_metadata_special_tokens_in_prefix: true
      metadata_prefix_sep: ' |||'
      metadata_prefix_start_seq: ''
      max_seq_len: 1024
      html_parser_config:
        all_tags_rules:
          attributes_to_keep:
          - class
          - id
          txt_max_chr_len: 0
          txt_min_chr_len: -.inf
          tags_exceptions_to_txt_max_min_chr_len:
          - table
          - tr
          - th
          - td
          - colgroup
          - thead
          - tfoot
          - tbody
        tags_to_remove_alone_tag_name:
        - body
        tags_to_remove_alone_txt_max_chr_len:
        - .inf
        tags_to_remove_alone_txt_min_chr_len:
        - 0.0
      local_metadata_special_token_start:
          entity_paragraph: "<ENTITY_CHAIN>"
          html: "<HTML>"
      local_metadata_special_token_end:
          entity_paragraph: " </ENTITY_CHAIN> "
          html: "</HTML>"
      local_metadata_special_token_state: true
      html_overall_sample_rate: 0.5
      without_metadata_same_context: false
    """

    data_config = yaml.safe_load(data_config_yaml)
    x = 1
    # data_with_entities = get_dataset(files_with_entities, num_gpus=0, gpu_id=None, data_config, tokenizer=)
