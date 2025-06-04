#!/bin/bash
N_TEST_SCENES=$1
OUTPUT_DIR=$2
ROOM_TYPE=$3
MODEL_CKPT=$4
DO_FULL_SCENES=$5

eval "$(conda shell.bash hook)"
conda deactivate
conda activate mi-diff

cd ./eval/baselines/mi-diff/MiDiffusion

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export __GLX_VENDOR_LIBRARY_NAME=nvidia

xvfb-run -a python ./midiff_custom_generate_scenes.py $MODEL_CKPT --config_file ./config/custom_config_${ROOM_TYPE}.yaml --path_to_pickled_3d_future_models ../ThreedFront/preprocessing-3dfuture-${ROOM_TYPE}.pkl --output_directory $OUTPUT_DIR --n-test-scenes=$N_TEST_SCENES --room-type $ROOM_TYPE $([ "$DO_FULL_SCENES" = true ] && echo "--do-full-scenes")