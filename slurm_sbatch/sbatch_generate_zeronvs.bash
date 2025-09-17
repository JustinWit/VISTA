#!/bin/bash
#SBATCH --job-name=VISTA
#SBATCH -p kira-lab
#SBATCH -G l40s:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH --array 0-4
#SBATCH -x hal,friday,irona

TASK=$1

VALID_MODEL=("zeronvs_mimicgen_ft")
MODEL="zeronvs_mimicgen_ft"
if [[ ! " ${VALID_MODEL[@]} " =~ " ${MODEL} " ]]; then
    echo "Error: MODEL must be one of: ${VALID_MODEL[*]}"
    exit 1
fi

if [[ "$MODEL" == "zeronvs_lpips_guard" ]]; then
    MODEL_NAME="zeronvs"
else
    MODEL_NAME="mimicgen_ft"
fi


set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate zeronvs

python VISTA/robomimic/robomimic/scripts/dataset_states_to_obs_zeronvs.py --dataset datasets/arc_90deg/$TASK/image_200.hdf5 --output_name random_cam_${MODEL_NAME}_domainB.hdf5  --done_mode 2 --randomize_cam_range arc_90deg --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --compress --exclude-next-obs --randomize_cam --parse-iters 1 --camera_randomization_type $MODEL --n 20 --start-idx $(((SLURM_ARRAY_TASK_ID + 5) * 20)) --visual_domain B
