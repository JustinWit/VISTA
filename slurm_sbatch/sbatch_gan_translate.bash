#!/bin/bash
#SBATCH --job-name=translate
#SBATCH -p kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 7
#SBATCH --qos=short
#SBATCH -x hal,friday,irona
#SBATCH --array 0-5

DATA=$1
CHECKPOINT=$2

VALID_TASK=("can" "coffee" "hammer" "square" "stack" "threading")
TASK=${VALID_TASK[$SLURM_ARRAY_TASK_ID]}

if [[ "$CHECKPOINT" == *"256"* ]]; then
    ARGS="--resize 256"
else
    ARGS=""
fi


set -ex
nvidia-smi

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate demo_translate

python -u demo_translate/translate_robomimic_sim.py --hdf5_file datasets/arc_90deg/$TASK/$DATA.hdf5 --name $CHECKPOINT --checkpoints_dir /coc/testnvme/jcoholich3/demo_translate/checkpoints $ARGS
