#!/bin/bash

N_TEST_SCENES=500

BON_LLM=1

export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# ************************************************************************************************************************************************************************************
# bedroom

ROOM_TYPE=bedroom
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --n-test-scenes=$N_TEST_SCENES --do-metrics

# ************************************************************************************************************************************************************************************
# livingroom

ROOM_TYPE=livingroom
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --n-test-scenes=$N_TEST_SCENES --do-metrics

# ************************************************************************************************************************************************************************************
# all

ROOM_TYPE=all
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/instr/${ROOM_TYPE}-bon${BON_LLM}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --use-vllm --do-bon-rotation

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env=".env" --room-type=$ROOM_TYPE --n-test-scenes=$N_TEST_SCENES --do-metrics