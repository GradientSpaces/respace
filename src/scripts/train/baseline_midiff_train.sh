#!/bin/bash

eval "$(conda shell.bash hook)"
conda deactivate
conda activate mi-diff

export __GLX_VENDOR_LIBRARY_NAME=nvidia

# ************************************************************************************************************************
# preprocess

cd ./eval/baselines/mi-diff/ThreedFront

# rm -f ./preprocessing-3dfront.pkl
# python scripts/pickle_threed_front_dataset.py ../../../../data/3D-FRONT ../../../../data/3D-FUTURE-assets ../../../../data/3D-FUTURE-assets/model_info.json --output_path ./preprocessing-3dfront.pkl

# # preprocess bedroom
# rm -f ./preprocessing-3dfuture-bedroom.pkl
# python scripts/pickle_threed_future_dataset.py no-filtering --output_path ./preprocessing-3dfuture-bedroom.pkl --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/bedroom_splits.csv
# rm -rf ./preprocessing-bedroom
# xvfb-run -a python scripts/preprocess_data.py no-filtering --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/bedroom_splits.csv --output_directory ./preprocessing-bedroom --room_side 8.0 --camera_position "0,10,0" --no_texture
# python scripts/preprocess_floorplan.py ./preprocessing-bedroom --room_side 8.0
# find /home/martinbucher/git/stan-24-sgllm/eval/baselines/mi-diff/ThreedFront/preprocessing-bedroom/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} find {} -maxdepth 1 -name "boxes.npz" | wc -l

# # preprocess livingroom
# rm -f ./preprocessing-3dfuture-livingroom.pkl
# python scripts/pickle_threed_future_dataset.py no-filtering --output_path ./preprocessing-3dfuture-livingroom.pkl --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/livingroom_splits.csv
# rm -rf ./preprocessing-livingroom
# xvfb-run -a python scripts/preprocess_data.py no-filtering --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/livingroom_splits.csv --output_directory ./preprocessing-livingroom --room_side 9.0 --camera_position "0,10,0" --no_texture
# find /home/martinbucher/git/stan-24-sgllm/eval/baselines/mi-diff/ThreedFront/preprocessing-livingroom/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} find {} -maxdepth 1 -name "boxes.npz" | wc -l
# python scripts/preprocess_floorplan.py ./preprocessing-livingroom --room_side 9.0

# preprocess all
rm -f ./preprocessing-3dfuture-all.pkl
python scripts/pickle_threed_future_dataset.py no-filtering --output_path ./preprocessing-3dfuture-all.pkl --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/all_splits.csv
rm -rf ./preprocessing-all
xvfb-run -a python scripts/preprocess_data.py no-filtering --path_to_pickled_3d_front_dataset ./preprocessing-3dfront.pkl --annotation-file ../../../../data/3D-FRONT-martin-rooms-stage-3/all_splits.csv --output_directory ./preprocessing-all --room_side 10.0 --camera_position "0,10,0" --no_texture
find /home/martinbucher/git/stan-24-sgllm/eval/baselines/mi-diff/ThreedFront/preprocessing-all/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} find {} -maxdepth 1 -name "boxes.npz" | wc -l
python scripts/preprocess_floorplan.py ./preprocessing-all --room_side 10.0

# ************************************************************************************************************************
# train

cd ../MiDiffusion/scripts

# # train bedroom
# rm -rf ../train-bedroom/apr14-bedroom
# python ./train_diffusion.py ../config/custom_config_bedroom.yaml --output_directory ../train-bedroom/ --experiment_tag "apr14-bedroom" --with_wandb_logger

# # train livingroom
# rm -rf ../train-livingroom/apr28-livingroom
# python ./train_diffusion.py ../config/custom_config_livingroom.yaml --output_directory ../train-livingroom/ --experiment_tag "apr28-livingroom" --with_wandb_logger

# # train all
rm -rf ../train-all/apr28-all
python ./train_diffusion.py ../config/custom_config_all.yaml --output_directory ../train-all/ --experiment_tag "apr28-all" --with_wandb_logger