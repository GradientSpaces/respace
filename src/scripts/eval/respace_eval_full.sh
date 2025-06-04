#!/bin/bash

N_TEST_SCENES=500

BON_LLM=8

DO_ICL_FOR_PROMPT=true
DO_CLASS_LABELS_FOR_PROMPT=true
DO_PROP_SAMPLING_FOR_PROMPT=true
ICL_K=2

export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# ************************************************************************************************************************************************************************************
# bedroom

ROOM_TYPE=bedroom
MODEL_ID=64663807/checkpoint-best # qwen1.5B full all + grpo beta 0.0 (may04)

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz

if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
    ICL_FLAG="--do-icl-for-prompt"
else
    ICL_FLAG=""
fi

if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
    CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
else
    CLASS_LABELS_FLAG=""
fi

if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
    PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
else
    PROP_SAMPLING_FLAG=""
fi

generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --do-bedroom-testset

compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# livingroom

ROOM_TYPE=livingroom
MODEL_ID=64663807/checkpoint-best # qwen1.5B full all + grpo beta 0.0 (may04)

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz

if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
    ICL_FLAG="--do-icl-for-prompt"
else
    ICL_FLAG=""
fi

if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
    CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
else
    CLASS_LABELS_FLAG=""
fi

if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
    PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
else
    PROP_SAMPLING_FLAG=""
fi

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --do-livingroom-testset

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# all

ROOM_TYPE=all
MODEL_ID=64663807/checkpoint-best # qwen1.5B full all + grpo beta 0.0 (may04)

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz

if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
    ICL_FLAG="--do-icl-for-prompt"
else
    ICL_FLAG=""
fi

if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
    CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
else
    CLASS_LABELS_FLAG=""
fi

if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
    PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
else
    PROP_SAMPLING_FLAG=""
fi

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene