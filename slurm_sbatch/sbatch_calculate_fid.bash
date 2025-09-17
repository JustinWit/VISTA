#!/bin/bash
#SBATCH --job-name=FID
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-5


VALID_TASK=("can" "coffee" "hammer" "square" "stack" "threading")
TASK=${VALID_TASK[$SLURM_ARRAY_TASK_ID]}

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/compute_fid_robomimic.py  --task $TASK --fake $1
