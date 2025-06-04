#!/bin/bash

#SBATCH --job-name=flashattn
#SBATCH --output=flashattn_tar_%j.out
#SBATCH --error=flashattn_tar_%j.err
#SBATCH --time=23:59:00
#SBATCH --mem=96G
#SBATCH --mail-type=FAIL

ml load python/3.9.0
ml load cuda/12.1.1
ml load gcc/10.3.0
ml load system nvtop

cd $HOME/stan-24-sgllm

python3.9 -m venv ./.venv
source .venv/bin/activate

MAX_JOBS=4 pip install flash-attn --no-build-isolation