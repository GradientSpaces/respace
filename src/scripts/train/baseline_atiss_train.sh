#!/bin/bash

eval "$(conda shell.bash hook)"
conda deactivate
conda activate atiss

cd ./eval/baselines/ATISS/scripts

# ls -alt ./eval/baselines/ATISS/preprocessing-bedroom
# ls -1 ./eval/baselines/ATISS/preprocessing-bedroom | wc -l
# ./eval/baselines/ATISS/preprocessing-bedroom/dataset_stats.txt

export __GLX_VENDOR_LIBRARY_NAME=nvidia

# ************************************************************************************************************************
# preprocess + train

## preprocess bedroom
# rm -rf ../preprocessing-bedroom
# xvfb-run -a python preprocess_data.py ../preprocessing-bedroom ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json ../demo/floor_plan_texture_images --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/bedroom_splits.csv --dataset_filtering=no-filtering --room_side=8.0 --camera_position="0,10,0"
# python pickle_threed_future_dataset.py ../preprocessing-bedroom ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/bedroom_splits.csv --dataset_filtering=no-filtering
# python train_network.py ../config/custom_config_bedroom.yaml ../train-bedroom/ --with_wandb_logger

# preprocess livingroom
# rm -rf ../preprocessing-livingroom
# xvfb-run -a python preprocess_data.py ../preprocessing-livingroom ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json ../demo/floor_plan_texture_images --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/livingroom_splits.csv --dataset_filtering=no-filtering --room_side=9.0 --camera_position="0,10,0"
# PATH_TO_SCENES="/tmp/threed_front.pkl" python pickle_threed_future_dataset.py ../preprocessing-livingroom ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/livingroom_splits.csv --dataset_filtering=no-filtering
# python train_network.py ../config/custom_config_livingroom.yaml ../train-livingroom/ --with_wandb_logger

# preprocess all
rm -rf ../preprocessing-all
PATH_TO_SCENES="/tmp/threed_front.pkl" xvfb-run -a python preprocess_data.py ../preprocessing-all ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json ../demo/floor_plan_texture_images --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/all_splits.csv --dataset_filtering=no-filtering --room_side=10.0 --camera_position="0,10,0"
PATH_TO_SCENES="/tmp/threed_front.pkl" python pickle_threed_future_dataset.py ../preprocessing-all ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json --annotation_file=../../../../data/3D-FRONT-martin-rooms-stage-3/all_splits.csv --dataset_filtering=no-filtering
python train_network.py ../config/custom_config_all.yaml ../train-all/ --with_wandb_logger