#!/usr/bin/bash

rm -f /Volumes/apollo11/data/invalid_threed_front_rooms.txt

cp /Volumes/apollo11/data/invalid_threed_front_rooms_original.txt /Volumes/apollo11/data/invalid_threed_front_rooms.txt

# python src/preprocessing/3d-front/03_extract_corners_for_rooms.py

# python src/preprocessing/3d-front/04_training_dataset_stage_1_json.py

# python src/preprocessing/3d-front/05_training_dataset_stage_2_prune.py