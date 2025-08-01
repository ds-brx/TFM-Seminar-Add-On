#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH --gpus 1
#SBATCH -o logs/%x.%N.%A.%a.out
#SBATCH -e logs/%x.%N.%A.%a.errors
#SBATCH -J DFT-ABL
#SBATCH --mail-type=END,FAIL
#SBATCH -a 1-4 

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with task ID $SLURM_ARRAY_TASK_ID";

export PYTHONUNBUFFERED=1

DATASETS=('ames' 'chess' 'parking' 'folktables')

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID-1]}

echo "Running DFT on $DATASET..."
python -u -m ablations --dataset "$DATASET" \
       --automl
       --max_configs 10

echo "DONE with dataset $DATASET";
echo "Finished at $(date)";