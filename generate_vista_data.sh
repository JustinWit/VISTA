#!/bin/bash
#SBATCH --job-name=vista_data_generation
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --array=0-4
#SBATCH --qos=short
#SBATCH -x hal,friday,irona

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs
cd vipl

python baseline_vista.py --data_path $1 -i $SLURM_ARRAY_TASK_ID