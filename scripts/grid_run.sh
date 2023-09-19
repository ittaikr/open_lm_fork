#!/bin/bash
# cd scripts
# Define arrays of parameter values
EPOCHS_VALUES=(8 16 32 64)
LR_VALUES=("3e-3" "15e-4" "75e-5" "3e-4")
LR_SCHEDULER_VALUES=("const" "cosine")

# Iterate over EPOCHS, LR, and LR_SCHEDULER combinations
for LR_SCHEDULER in "${LR_SCHEDULER_VALUES[@]}"; do
  if [ "$LR_SCHEDULER" == "const" ]; then
    EPOCHS=64
    for LR in "${LR_VALUES[@]}"; do
      # Construct the export string
      EXPORT_STRING="EPOCHS_ENV=$EPOCHS,LR_ENV=$LR,LR_SCHEDULER_ENV=$LR_SCHEDULER"
      JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER"
      # Run sbatch with the constructed export string
      sbatch --job-name=$JOB_STR --output=logs_grid/%x_%j.out --error=logs_grid/%x_%j.out scripts/jsc_script.sh $EPOCHS $LR $LR_SCHEDULER
    done
  elif [ "$LR_SCHEDULER" == "cosine" ]; then
    LR="3e-3"
    for EPOCHS in "${EPOCHS_VALUES[@]}"; do
      # Construct the export string
      EXPORT_STRING="EPOCHS_ENV=$EPOCHS,LR_ENV=$LR,LR_SCHEDULER_ENV=$LR_SCHEDULER"
      JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER"
      # Run sbatch with the constructed export string
      sbatch --job-name=$JOB_STR --output=logs_grid/%x_%j.out --error=logs_grid/%x_%j.out scripts/jsc_script.sh $EPOCHS $LR $LR_SCHEDULER
      
    done
  fi
done
