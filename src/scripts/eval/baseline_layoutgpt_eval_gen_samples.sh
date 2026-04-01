#!/bin/bash
N_TEST_SCENES=$1
OUTPUT_DIR=$2
ROOM_TYPE=$3

eval "$(conda shell.bash hook)"
conda deactivate
conda activate layoutgpt

cd ./eval/baselines/LayoutGPT

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# xvfb-run -a python ./layoutgpt_custom_generate_scenes.py --output_directory=$OUTPUT_DIR --n-test-scenes=$N_TEST_SCENES --room-type=$ROOM_TYPE --gpt_type "gpt-4o" --K 8 --temperature 0.7 --path_to_pickled_3d_future_models="/home/martinbucher/git/stan-24-sgllm/eval/baselines/ATISS/preprocessing-all/threed_future_model_no-filtering.pkl" --atiss_config_file="../ATISS/config/custom_config_${ROOM_TYPE}.yaml" --unit px --normalize --max_train_examples 100 # --verbose --force_reload_cache
xvfb-run -a python ./layoutgpt_custom_generate_scenes_v2.py --output_directory=$OUTPUT_DIR --n-test-scenes=$N_TEST_SCENES --room-type=$ROOM_TYPE --gpt_type "gpt-4o" --K 8 --temperature 0.7 --path_to_pickled_3d_future_models="/home/martinbucher/git/stan-24-sgllm/eval/baselines/ATISS/preprocessing-all/threed_future_model_no-filtering.pkl" --atiss_config_file="../ATISS/config/custom_config_${ROOM_TYPE}.yaml" --unit px --normalize --max_train_examples 100 --verbose # --force_reload_cache