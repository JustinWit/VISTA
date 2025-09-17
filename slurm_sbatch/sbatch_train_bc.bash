#!/bin/bash
#SBATCH --job-name=BC_train
#SBATCH -p kira-lab
#SBATCH -G a40:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-2

VALID_TASK=("can" "coffee" "hammer" "square" "stack" "threading")
VALID_MODE=("deproj" "mimicgen_ft" "nvs" "orig" "sim" "untranslated" "translated")


TASK=$1
VALID_TASK=("can" "coffee" "hammer" "square" "stack" "threading")

MODE=$2
if [[ ! " ${VALID_MODE[@]} " =~ " ${MODE} " ]]; then
    echo "Error: MODE must be one of: ${VALID_MODE[*]}"
    exit 1
fi

DATASET=$3
SAMPLER=$4

filename="${DATASET##*/}"   # remove path → random_cam_sim_domainA_translated_by_sim_upscaled_256__0.hdf5
filename="${filename%.*}" # remove extension

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/robomimic/robomimic/scripts/train.py --config VISTA/robomimic/robomimic/exps/sim2sim/$TASK/$MODE.json --run_number ${SLURM_ARRAY_TASK_ID}_${filename}_${SAMPLER}_rerun400 --seed $((SLURM_ARRAY_TASK_ID + 1)) --dataset $DATASET --test_distro $SAMPLER
# python -u VISTA/robomimic/robomimic/scripts/train.py --config VISTA/robomimic/robomimic/exps/sim2sim/$TASK/$MODE.json --run_number ${SLURM_ARRAY_TASK_ID}_$SAMPLER --seed $((SLURM_ARRAY_TASK_ID + 1)) --test_distro $SAMPLER