#!/bin/bash
#
#SBATCH --job-name=AdptCtrl
#SBATCH --mail-user=dimarco.a@di.uniroma1.it
#SBATCH --mail-type=ALL
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err

INDEX=${SLURM_ARRAY_TASK_ID}

singularity exec --bind ../app:/app \
                 --bind ../data:/data \
                 image.sif \
                 python3 /app/main.py

