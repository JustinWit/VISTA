#!/bin/bash
#SBATCH --job-name=gan_data
#SBATCH -p kira-lab
#SBATCH -G titan_x:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array=

TASK=$1

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

# Figure out which command and repeat
START_IDX=$(( SLURM_ARRAY_TASK_ID * 50 ))

python -u VISTA/robomimic/robomimic/scripts/generate_robosuite_nvs_data.py --dataset datasets/train_vista/$TASK.hdf5 --output_name gan_data_variedB/${TASK}_10.hdf5  --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs --camera_randomization_type sim --views_per_state 10 --include_seg --depth  --visual_domain A
