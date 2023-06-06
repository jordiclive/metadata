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

    # if len(file_paths) >= num_gpus:
    #     data = data.shard(num_gpus, gpu_id)
    # add shuffle buffer size equal to the number of files, so each epoch is a different shuffle
    buffer_size = len(file_paths)
    data = data.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=True)
    data = data.interleave(
        lambda x: tf.data.TextLineDataset(x, compression_type="GZIP"),
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=4,
        block_length=16,
    )
    # Reads 4 files concurrently, and interleave blocks of 16 lines

    # if len(file_paths) < num_gpus:
    #     data = data.shard(num_gpus, gpu_id)

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


    class Args:
        class DataConfig:
            class MetadataConfig:
                class HtmlParserConfig:
                    class AllTagsRules:
                        def __init__(self):
                            self.attributes_to_keep = ['class', 'id']
                            self.txt_max_chr_len = 0
                            self.txt_min_chr_len = float('-inf')
                            self.tags_exceptions_to_txt_max_min_chr_len = ['table', 'tr', 'th', 'td', 'colgroup',
                                                                           'thead', 'tfoot', 'tbody']

                    def __init__(self):
                        self.all_tags_rules = self.AllTagsRules()
                        self.tags_to_remove_alone_tag_name = ['body']
                        self.tags_to_remove_alone_txt_max_chr_len = [float('inf')]
                        self.tags_to_remove_alone_txt_min_chr_len = [0.0]

                def __init__(self):
                    self.random_sample_metadata = True
                    self.random_sample_metadata_calculate_size = 16384
                    self.random_sample_metadata_weights = {
                        'html': 1.0,
                        'timestamp': 11.335125448028673,
                        'website_desc': 10.532889258950874,
                        'title': 1.0657717366883845,
                        'generation_datasource': 1.0,
                        'entity_paragraph': 1.028817740667444,
                        'generation_length_text': 1.0,
                    }
                    self.metadata_list = ['html', 'timestamp', 'website_description', 'title', 'url', 'datasource',
                                          'length', 'entity_paragraph', 'generation_length_text']
                    self.metadata_column_list = ['html', 'timestamp', 'website_desc', 'title', 'generation_datasource',
                                                 'generation_length_text', 'entity_paragraph']
                    self.local_metadata_special_tokens = {'entity_paragraph': "entity"}
                    self.metadata_sep = ' | '
                    self.metadata_key_value_sep = ': '
                    self.metadata_probability = 0.5
                    self.treat_local_metadata_as_regular_text = True
                    self.add_local_metadata_special_tokens_in_prefix = True
                    self.metadata_prefix_sep = ' |||'
                    self.metadata_prefix_start_seq = ''
                    self.max_seq_len = 1024
                    self.html_parser_config = self.HtmlParserConfig()
                    self.local_metadata_special_token_start = {
                        'entity_paragraph': "<ENTITY_CHAIN>",
                        'html': "<HTML>",
                    }
                    self.local_metadata_special_token_end = {
                        'entity_paragraph': " </ENTITY_CHAIN> ",
                        'html': "</HTML>",
                    }
                    self.local_metadata_special_token_state = True
                    self.html_overall_sample_rate = 0.5
                    self.without_metadata_same_context = False

            def __init__(self):
                self.streaming = True
                self.validation_size_max = 1024
                self.use_full_evaluation_for_val = True
                self.metadata_config = self.MetadataConfig()
                self.experiment = 'with_metadata_datasetv2_tf'
                self.per_device_eval_batch_size = 32
                self.per_device_train_batch_size = 32
                self.dataset_name = 'bs-modeling-metadata/c4-en-html-with-training_metadata_all'
                self.dataset_config_name = None
                self.train_file = 'c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz'
                self.validation_file = None
                self.overwrite_cache = False
                self.cache_dir = None
                self.extension = None
                self.preprocessing_num_workers = 40
                self.validation_split_percentage = 5
                self.block_size = None
                self.map_batch_size = 1

        def __init__(self):
            self.data_config = self.DataConfig()
            self.weight_decay = 0.01
            self.learning_rate = 0.0001
            self.wb_name = "all_metadata"
            self.num_train_epochs = 1
            self.max_train_steps = 100000
            self.lr_scheduler_type = "linear"
            self.num_warmup_steps = 6000
            self.seed = 42
            self.out_dir = "metadata_outputs/"
            self.model_name = "gpt2"
            self.project_name = "metadata_lm"
            self.jobid = ''
            self.start_with_eval = False
            self.extra_steps_to_eval_save_at = [2]
            self.evaluation_strategy = "STEPS"
            self.eval_num_per_epoch = 3
            self.eval_steps = 250
            self.save_strategy = "STEPS"
            self.save_num_per_epoch = 3
            self.save_steps = 1000
            self.do_train = True
            self.do_eval = True
            self.gradient_checkpointing = True
            self.resume_from_checkpoint_dir = None
            self.gradient_accumulation_steps = 2
    #
    # class DataConfig:
    #     class MetadataConfig:
    #         class HtmlParserConfig:
    #             class AllTagsRules:
    #                 def __init__(self):
    #                     self.attributes_to_keep = ['class', 'id']
    #                     self.txt_max_chr_len = 0
    #                     self.txt_min_chr_len = float('-inf')
    #                     self.tags_exceptions_to_txt_max_min_chr_len = ['table', 'tr', 'th', 'td', 'colgroup', 'thead',
    #                                                                    'tfoot', 'tbody']
    #
    #             def __init__(self):
    #                 self.all_tags_rules = self.AllTagsRules()
    #                 self.tags_to_remove_alone_tag_name = ['body']
    #                 self.tags_to_remove_alone_txt_max_chr_len = [float('inf')]
    #                 self.tags_to_remove_alone_txt_min_chr_len = [0.0]
    #
    #         def __init__(self):
    #             self.random_sample_metadata = True
    #             self.random_sample_metadata_calculate_size = 16384
    #             self.random_sample_metadata_weights = {
    #                 'html': 1.0,
    #                 'timestamp': 11.335125448028673,
    #                 'website_desc': 10.532889258950874,
    #                 'title': 1.0657717366883845,
    #                 'generation_datasource': 1.0,
    #                 'entity_paragraph': 1.028817740667444,
    #                 'generation_length_text': 1.0
    #             }
    #             self.metadata_list = ['html', 'timestamp', 'website_description', 'title', 'url', 'datasource',
    #                                   'length', 'entity_paragraph', 'generation_length_text']
    #             self.metadata_column_list = ['html', 'timestamp', 'website_desc', 'title', 'generation_datasource',
    #                                          'generation_length_text', 'entity_paragraph']
    #             self.html_parser_config = self.HtmlParserConfig()
    #
    #     def __init__(self):
    #         self.streaming = True
    #         self.validation_size_max = 1024
    #         self.use_full_evaluation_for_val = True
    #         self.metadata_config = self.MetadataConfig()


    # class Args:
    #     def __init__(self):
    #         self.data_config = DataConfig()
    #         self.weight_decay = 0.01
    #         self.learning_rate = 0.0001
    #         self.wb_name = "all_metadata"
            # Rest of the properties should be added here


    args = Args()
    print(args.data_config.metadata_config.metadata_column_list)
    data_config = args.data_config
    data_config.metadata_config.entity_setting = 'beg'
    print(data_config.metadata_config.metadata_column_list)
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("tokenizer")
    x = 1
    dataset = [Path("/Users/jordanclive/Personal_git/metadata/local-data/datasets--bs-modeling-metadata--c4-en-html-with-training_metadata_all/snapshots/8f2615d8b8580e89533b90bc3931e0b99ef15aec/c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz")]
    data_with_entities = get_dataset(dataset, num_gpus=0, gpu_id=0, data_config=data_config, tokenizer=tokenizer)
    data = tf.data.Dataset.sample_from_datasets(
        [data_with_entities],
        weights=[float(len(dataset))],
        seed=42,
    )

    data = data.shuffle(1000, reshuffle_each_iteration=True)
    data = data.batch(8)
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


    while True:
        for batch in data:
            batch = to_dict(batch)
            x =1
