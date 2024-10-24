#!/bin/bash -x

#SBATCH --account=transfernetx
# SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# SBATCH --time=30:00
#SBATCH --time=6:00:00
#SBATCH --partition=booster
# SBATCH --partition=develbooster
#SBATCH --job-name=traverse_eval
#SBATCH --output=logs_traverse/%x_%j.out
#SBATCH --error=logs_traverse/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tomerporian@mail.tau.ac.il
#SBATCH --mem=64G

ml purge

CONDA_ENV="py9"
source /p/project1/ccstdl/porian1/miniconda3/bin/activate ${CONDA_ENV}

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export NCCL_ASYNC_ERROR_HANDLING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"

OPEN_CLIP_HOME="/p/project1/ccstdl/$USER/open_lm_fork"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"
cd ${OPEN_CLIP_HOME}

WANDB_MODE=offline


# srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python traverse_eval_bot.py 
CUDA_VISIBLE_DEVICES=0 python traverse_eval_bot.py &
sleep 5
CUDA_VISIBLE_DEVICES=1 python traverse_eval_bot.py &
sleep 5
CUDA_VISIBLE_DEVICES=2 python traverse_eval_bot.py &
sleep 5
CUDA_VISIBLE_DEVICES=3 python traverse_eval_bot.py
sleep 5