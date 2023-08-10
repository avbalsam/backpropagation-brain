#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
##SBATCH --gres=gpu:1 # number of GPUs
##SBATCH --constraint=high-capacity
##SBATCH --constraint=32GB
##SBATCH --mem=16GB
#SBATCH --array=1
#SBATCH -D /sci/home/forkosh/git/backpropagation-brain/logs
cd /sci/home/forkosh/git/backpropagation-brain
hostname
date "+%y/%m/%d %H:%M:%S"
# module load openmind/singularity/3.4.1
# module add openmind/cuda/11.3
# module add openmind/cudnn/11.5-v8.3.3.40
# source /home/jangh/.bashrc
# conda activate openmind
module load python
python ./Code/chessboard_random_walk.py
# singularity exec --nv -B /om,/om2 /om2/user/xboix/singularity/xboix-tensorflow2.9b.simg \
# python prednet/mnist_train.py \
# --is_slurm=True \
# --job=${SLURM_ARRAY_JOB_ID} \
# --id=${SLURM_ARRAY_TASK_ID} \