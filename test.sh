#!/bin/bash

source /home/jordan/miniconda3/envs/jordan_metadata_312/bin/activate
conda activate jordan_metadata_312
# source /home/jordan/miniconda3/envs/jordan_metadata_dev/bin/activate
# conda activate jordan_metadata_dev

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache

#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
#export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export MODEL=gpt2-xl #google/gemma-2b
export NUM_GPU=1
export DEEPSPEED_CONFIG=$(realpath /home/jordan/MRs/metadata/bsmetadata/deepspeed_configs/v2.json)
export DATA_DIR=$(realpath /home/jordan/MRs/metadata/local_data/)
echo "deepspeed_config_file: $DEEPSPEED_CONFIG"

export WANDB_API_KEY= 'd8216641d549f9bb3d0c5074baa39e15dfd55030'

echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 1
" > accelerate_config.yaml
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py --config-name v2 \
  model_name=$MODEL \
  data_config.dataset_name=$DATA_DIR \
  data_config.train_file='*.jsonl.gz' \
  data_config.validation_file='c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz' \
  out_dir=/home/jordan/MRs/metadata/output_folder \
  wb_name="full_retest" 2>&1 | tee /home/jordan/MRs/metadata/train_output.log
# mixed_precision: fp16
#fp16: true

#  wb_name="full-metadata-with-generation-text-0.5-html"