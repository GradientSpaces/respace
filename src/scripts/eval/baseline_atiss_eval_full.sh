#!/bin/bash

N_TEST_SCENES=500
DO_FULL_SCENES=true

# ************************************************************************************************************************************************************************************
# bedroom

ROOM_TYPE=bedroom
MODEL_CKPT=../train-bedroom/S2WWC30OY/model_best.pth

OUTPUT_DIR_SCENES=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/json
OUTPUT_DIR_VIZ=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
./src/scripts/eval/baseline_atiss_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# evaluate samples
source .venv/bin/activate
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# livingroom

# ROOM_TYPE=livingroom
# MODEL_CKPT=../train-livingroom/34OHAO7UR/model_best.pth

# OUTPUT_DIR_SCENES=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/viz

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678
# ./src/scripts/eval/baseline_atiss_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# # evaluate samples
# source .venv/bin/activate
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# all

# ROOM_TYPE=all
# MODEL_CKPT=../train-all/6G41AXKKZ/model_best.pth

# OUTPUT_DIR_SCENES=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-atiss/full/${ROOM_TYPE}/viz

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678
# ./src/scripts/eval/baseline_atiss_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# # evaluate samples
# source .venv/bin/activate
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene