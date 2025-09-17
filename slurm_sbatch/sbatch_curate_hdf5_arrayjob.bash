#!/bin/bash
#SBATCH --job-name=generate_sim
#SBATCH -p kira-lab
#SBATCH -G titan_x:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-39

ROOT=$1
OUTPUT=$2
PREFIX=$3

ALL_FILES=("$ROOT"/*.hdf5)

task_id=$(( SLURM_ARRAY_TASK_ID % 8 ))
file=${ALL_FILES[$task_id]}

repeat=$(( SLURM_ARRAY_TASK_ID / 8 ))
start_idx=$(( repeat * 30 ))

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/robomimic/robomimic/scripts/curateGANdata_arrayjob.py --root $ROOT --file $file --output $OUTPUT --prefix $PREFIX --n 30 --start-idx $start_idx --views_per_frame 1