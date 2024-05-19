#!/bin/bash
# training on the stronger GPU's in the killable partition
#SBATCH --partition=killable
#SBATCH --time=1-0:00:00       # max time (minutes)
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=32G               # memory
#SBATCH --cpus-per-task=8      # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --constraint="tesla_v100|geforce_rtx_3090|a5000"
source ~/.bashrc
conda activate dev


LOGS="./$3"
WANDB_MODE=offline

python -u -m open_lm.main --name "$2" --logs $LOGS --config $1