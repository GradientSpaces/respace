#!/bin/bash
N_TEST_SCENES=$1
OUTPUT_DIR=$2
ROOM_TYPE=$3

eval "$(conda shell.bash hook)"
conda deactivate
conda activate layoutvlm

cd ./eval/baselines/LayoutVLM

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export __GLX_VENDOR_LIBRARY_NAME=nvidia

xvfb-run -a python layoutvlm_custom_generate.py --output_directory=$OUTPUT_DIR --room-type=$ROOM_TYPE --n-test-scenes=$N_TEST_SCENES