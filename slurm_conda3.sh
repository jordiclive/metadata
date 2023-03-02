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


echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID




echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
fp16: true
main_training_function: main
mixed_precision: fp16
" > accelerate_config.yaml


export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


srun accelerate launch --config_file accelerate_config.yaml --num_processes $(( 8 * $COUNT_NODE )) --num_machines $COUNT_NODE --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  bsmetadata/train.py --config-name v2 \
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



