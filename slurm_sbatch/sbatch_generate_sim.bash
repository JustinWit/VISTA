#!/bin/bash
#SBATCH --job-name=generate_sim
#SBATCH -p kira-lab
#SBATCH -G titan_x:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona

TASK=$1
DOMAIN=$2

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python VISTA/robomimic/robomimic/scripts/dataset_states_to_obs_zeronvs.py --dataset datasets/arc_90deg/$TASK/image_200.hdf5 --output_name random_cam_sim_domain$DOMAIN.hdf5  --done_mode 2 --randomize_cam_range arc_90deg --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --compress --exclude-next-obs --randomize_cam --parse-iters 1 --camera_randomization_type sim --visual_domain $DOMAIN
