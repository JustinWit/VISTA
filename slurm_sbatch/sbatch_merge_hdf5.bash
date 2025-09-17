#!/bin/bash
#SBATCH --job-name=merge
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-7



# VALID_TASK=("coffee" "can" "square" "hammer" "stack" "threading")
VALID_TASK=("coffee_preparation_d1"  "kitchen_d1"  "mug_cleanup_d1"  "nut_assembly_d0"  "pick_place_d0"  "square_d2"  "stack_three_d1"  "three_piece_assembly_d2")

TASK=${VALID_TASK[SLURM_ARRAY_TASK_ID]}
# TASK="threading"

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python -u VISTA/robomimic/robomimic/scripts/merge_hdf5.py --input_files datasets/train_vista/simpler_env/$TASK/* --output_file  datasets/train_vista/simpler_env/${TASK}_0-150.hdf5
# python -u VISTA/robomimic/robomimic/scripts/merge_hdf5.py --input_files datasets/arc_90deg/$TASK/deproj/* --output_file  datasets/arc_90deg/$TASK/random_cam_deproj_domainB.hdf5