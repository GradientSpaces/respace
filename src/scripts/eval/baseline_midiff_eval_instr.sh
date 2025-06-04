#!/bin/bash

N_TEST_SCENES=500
DO_FULL_SCENES=false

# ************************************************************************************************************************************************************************************
# bedroom

ROOM_TYPE=bedroom
MODEL_CKPT=./train-bedroom/apr14-bedroom/best_model.pt

OUTPUT_DIR_SCENES=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/json
OUTPUT_DIR_VIZ=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
./src/scripts/eval/baseline_midiff_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# evaluate samples
source .venv/bin/activate
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES

# # ************************************************************************************************************************************************************************************
# livingroom

# ROOM_TYPE=livingroom
# MODEL_CKPT=./train-livingroom/apr28-livingroom/best_model.pt

# OUTPUT_DIR_SCENES=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/viz

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678
# ./src/scripts/eval/baseline_midiff_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# evaluate samples
# source .venv/bin/activate
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES

# # ************************************************************************************************************************************************************************************
# # all

# ROOM_TYPE=all
# MODEL_CKPT=./train-all/apr28-all/best_model.pt

# OUTPUT_DIR_SCENES=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-midiff/instr/${ROOM_TYPE}/viz

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678
# ./src/scripts/eval/baseline_midiff_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE $MODEL_CKPT $DO_FULL_SCENES

# # evaluate samples
# source .venv/bin/activate
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES