#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
source ~/miniconda3/bin/activate meta_conda
cd /admin/home-jordiclive/metadata
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1


export MODEL=gpt2-xl
export NUM_GPU=8

export DEEPSPEED_CONFIG=$(realpath bsmetadata/deepspeed_configs/v2.json)
export DATA_DIR=$(realpath /admin/home-jordiclive/local_data)
echo "deepspeed_config_file: $DEEPSPEED_CONFIG"

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR


echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
fp16: true
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $NUM_GPU
mixed_precision: fp16
" > accelerate_config.yaml


export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun bash experiments/hpsearch/test.sh

accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py --config-name v2 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  model_name=$MODEL \
    data_config.train_file='*.jsonl.gz' \
    data_config.validation_file='c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz' \
    data_config.dataset_name=$DATA_DIR \
    data_config.preprocessing_num_workers=6  extra_steps_to_eval_save_at='[2,100,200,400,800]' \
    data_config.metadata_config.metadata_list='[html]' \
    data_config.metadata_config.metadata_column_list='[html]' \
    out_dir=$HOME/tmp/metadata-run-html
    #out_dir=/mnt/ssd-1/bigscience-metadata/run1
    #data_config.train_file='c4-en-html_cc*.jsonl.gz' data_config.streaming=True out_dir=/mnt/ssd-1/bigscience-metadata/run1



