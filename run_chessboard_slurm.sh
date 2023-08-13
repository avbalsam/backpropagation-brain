#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
module load python/3.9.7

. /sci/labs/forkosh/forkosh/avi_venv/bin/activate

which python3

python3 ./Code/chessboard_random_walk.py
