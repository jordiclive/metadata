#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40x"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
source /admin/home-jordiclive/miniconda3/bin/activate jordan_meta
cd /fsx/home-jordiclive/metadata
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
export DS_SKIP_CUDA_CHECK=1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
#export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1


export MODEL=gpt2-xl
export NUM_GPU=8
export DEEPSPEED_CONFIG=$(realpath bsmetadata/deepspeed_configs/v2.json)
export DATA_DIR=$(realpath /fsx/home-jordiclive/metadata/local-data/)
echo "deepspeed_config_file: $DEEPSPEED_CONFIG"

export WANDB_API_KEY= 'd8216641d549f9bb3d0c5074baa39e15dfd55030'

echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: -1
mixed_precision: fp16
" > accelerate_config.yaml
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py --config-name v2 \
  model_name=$MODEL \
  data_config.dataset_name=$DATA_DIR \
  data_config.train_file='*.jsonl.gz' \
  data_config.validation_file='c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz' \
  out_dir=/fsx/home-jordiclive/tmp/metadata-html-half \
#  wb_name="full-metadata-with-generation-text-0.5-html"