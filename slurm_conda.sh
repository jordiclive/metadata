#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
source ~/miniconda3/bin/activate meta_conda

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1


export MASTER_PORT=8994


cd /admin/home-jordiclive/metadata
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="2,3"
srun CUDA_VISIBLE_DEVICES="2,3" bash experiments/hpsearch/test.sh