#!/bin/bash
#SBATCH --job-name=deproj
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-4

TASK=$1

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate 3dproj

python VISTA/robomimic/robomimic/scripts/dataset_states_to_obs_zeronvs.py --dataset datasets/arc_90deg/$TASK/image_200.hdf5 --output_name random_cam_deproj_domainB.hdf5  --done_mode 2 --randomize_cam_range arc_90deg --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --compress --exclude-next-obs --randomize_cam --parse-iters 1 --camera_randomization_type deproj --n 10 --start-idx $((SLURM_ARRAY_TASK_ID * 10))  --visual_domain B
# $((SLURM_ARRAY_TASK_ID * 50))
