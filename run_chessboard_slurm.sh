#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
module load python
python ./Code/chessboard_random_walk.py
# singularity exec --nv -B /om,/om2 /om2/user/xboix/singularity/xboix-tensorflow2.9b.simg \
# python prednet/mnist_train.py \
# --is_slurm=True \
# --job=${SLURM_ARRAY_JOB_ID} \
# --id=${SLURM_ARRAY_TASK_ID} \