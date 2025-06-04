#!/bin/bash
N_TEST_SCENES=$1
OUTPUT_DIR=$2
ROOM_TYPE=$3
MODEL_CKPT=$4
DO_FULL_SCENES=$5

eval "$(conda shell.bash hook)"
conda deactivate
conda activate atiss

cd ./eval/baselines/ATISS/scripts

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export __GLX_VENDOR_LIBRARY_NAME=nvidia

xvfb-run -a python ../atiss_custom_generate_scenes.py ../config/custom_config_${ROOM_TYPE}.yaml ../preprocessing-${ROOM_TYPE}/threed_future_model_no-filtering.pkl ../demo/floor_plan_texture_images --weight_file=$MODEL_CKPT --without_screen --output_directory=$OUTPUT_DIR --n-test-scenes=$N_TEST_SCENES --room-type=$ROOM_TYPE $([ "$DO_FULL_SCENES" = true ] && echo "--do-full-scenes")