#!/bin/bash
#SBATCH --job-name=real_vista
#SBATCH -p kira-lab
#SBATCH -G a40:1
#SBATCH -c 15
#SBATCH --qos=short
#SBATCH -x hal,friday,irona

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/vipl/baseline_vista.py --data_path $1 --n 10 --start-idx  0