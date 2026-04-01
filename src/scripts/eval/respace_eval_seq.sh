#!/bin/bash

N_TEST_SCENES=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# ************************************************************************************************************************************************************************************
# SEQ evals

ROOM_TYPE=all
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

BON_LLM=8

OUTPUT_DIR_SCENES=./eval/samples/respace/seq/${ROOM_TYPE}-bon${BON_LLM}-bonrot/json
OUTPUT_DIR_VIZ=./eval/samples/respace/seq/${ROOM_TYPE}-bon${BON_LLM}-bonrot/viz

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678

xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env=".env" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM --do-seq-test --use-vllm --do-bon-rotation