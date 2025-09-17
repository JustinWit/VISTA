#!/bin/bash
#SBATCH --job-name=gen_train
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-5

TASK=$1

set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

# Figure out start idx
START_IDX=$(( SLURM_ARRAY_TASK_ID * 15 ))

# This is fixed in domain cam which is domain B
# python -u VISTA/robomimic/robomimic/scripts/generate_robosuite_nvs_data.py --dataset datasets/arc_90deg/$TASK/image_200.hdf5 --output_name ../../train_vista/gan_data_fixedB_eval_tasks/${TASK}.hdf5  --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs --camera_randomization_type sim --views_per_state 1 --include_seg --depth --visual_domain B --n 15 --start-idx $START_IDX --compress

# This is varied out of domain cam 
# python -u VISTA/robomimic/robomimic/scripts/generate_robosuite_nvs_data.py --dataset datasets/train_vista/$TASK.hdf5 --output_name simpler_env/${TASK}/${TASK}.hdf5  --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs --camera_randomization_type sim --views_per_state 1 --include_seg --depth --randomize_cam --compress --visual_domain D --n 15 --start-idx $START_IDX 

# This is test data between B and D
python -u VISTA/robomimic/robomimic/scripts/generate_robosuite_nvs_data.py --dataset datasets/arc_90deg/$TASK/image_200.hdf5 --output_name ../../train_vista/simpler_env/testdata/${TASK}.hdf5  --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs --camera_randomization_type sim --views_per_state 1 --depth --include_seg --randomize_cam --visual_domain B --n 1 --start-idx $SLURM_ARRAY_TASK_ID  --test_data D --compress
# python -u VISTA/robomimic/robomimic/scripts/generate_robosuite_nvs_data.py --dataset datasets/train_vista/$TASK.hdf5 --output_name blackbg_texture_table/${TASK}.hdf5  --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs --camera_randomization_type sim --views_per_state 1 --depth --include_seg --randomize_cam --visual_domain C --n 1 --start-idx 0  --test_data D --compress
