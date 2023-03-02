#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40423"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_meta/bin/activate
cd /admin/home-jordiclive/metadata
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun bash experiments/hpsearch/test.sh