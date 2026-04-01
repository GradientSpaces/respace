#!/bin/bash

N_TEST_SCENES=200

# ************************************************************************************************************************************************************************************

# ROOM_TYPE=all
ROOM_TYPE=bedroom

# OUTPUT_DIR_SCENES=./eval/samples/baseline-layoutgpt/full/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-layoutgpt/full/${ROOM_TYPE}/viz

OUTPUT_DIR_SCENES=./eval/exp-latency/baseline-layoutgpt/full/${ROOM_TYPE}/json

# OUTPUT_DIR_SCENES=./eval/samples/baseline-layoutgpt-run2/full/${ROOM_TYPE}/json
# OUTPUT_DIR_VIZ=./eval/samples/baseline-layoutgpt-run2/full/${ROOM_TYPE}/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
./src/scripts/eval/baseline_layoutgpt_eval_gen_samples.sh $N_TEST_SCENES $OUTPUT_DIR_SCENES $ROOM_TYPE

# # evaluate samples
# source .venv/bin/activate
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --n-test-scenes=$N_TEST_SCENES --is-full-scene --do-rectangular-only # --do-metrics

# ************************************************************************************************************************************************************************************