#!/bin/bash

N_TEST_SCENES=500

BON_LLM=1
BON_SHUFFLING=8

export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# ************************************************************************************************************************************************************************************
# bedroom

ROOM_TYPE=bedroom
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --bon-shuffling=$BON_SHUFFLING --do-bon-shuffling --do-icl-for-prompt --do-class-labels-for-prompt --do-prop-sampling-for-prompt --icl-k=2 --do-full-scenes --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# livingroom

ROOM_TYPE=livingroom
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --bon-shuffling=$BON_SHUFFLING --do-bon-shuffling --do-icl-for-prompt --do-class-labels-for-prompt --do-prop-sampling-for-prompt --icl-k=2 --do-full-scenes --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# all

ROOM_TYPE=all
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-bon${BON_LLM}-s${BON_SHUFFLING}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --bon-shuffling=$BON_SHUFFLING --do-bon-shuffling --do-icl-for-prompt --do-class-labels-for-prompt --do-prop-sampling-for-prompt --icl-k=2 --do-full-scenes --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene
