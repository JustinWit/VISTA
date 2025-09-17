#!/bin/bash
#SBATCH --job-name=generate_sim
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona

ROOT=$1
OUTPUT=$2
PREFIX=$3

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/robomimic/robomimic/scripts/curateGANdata.py --root $ROOT --output $OUTPUT --prefix $PREFIX --test_data