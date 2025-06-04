#!/usr/bin/bash

#SBATCH --partition=serc
#SBATCH --nodes=1

#SBATCH --gpus=1

#SBATCH -C GPU_MEM:40GB

## B8: 23/bed, 47/liv, 35/all
## B4: 09/bed, 24/liv, 18/all

## B2: 27/bed, 39/all
## B4: 35/bed, 57/all 

#SBATCH --time=72:29:00

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=96G
#SBATCH --job-name=sgllm
#SBATCH --output=logs/%j/out.log
#SBATCH --error=logs/%j/err.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export N_TASKS=1

JOB_ID=$SLURM_JOB_ID
mkdir -p ./logs
mkdir -p "./logs/${JOB_ID}"
mkdir -p "${SCRATCH}/ckpts"
ln -sf "${SCRATCH}/ckpts" "./ckpts"

export HF_HOME="${SCRATCH}/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

ml load python/3.12.1
ml load cuda/12.4.0
ml load gcc/12.4.0
ml load system nvtop

cd $HOME/stan-24-sgllm
git pull
source .venv_py312/bin/activate

pip install torch accelerate importlib_metadata
pip install --upgrade transformers

# ************************************************************************
# python src/pipeline.py --use-gpu --pth-output="./eval/samples/respace/full/bedroom-with-qwen1.5b-all-bon-8/json" --env="sherlock" --jid="$(uuidgen)" --room-type="bedroom" --model-id="63794613/checkpoint-best" --n-test-scenes=500 --bon-llm=8 --do-icl-for-prompt --do-class-labels-for-prompt --do-prop-sampling-for-prompt --icl-k=2 --do-full-scenes --do-bedroom-testset

N_TEST_SCENES=500
DO_ICL_FOR_PROMPT=true
DO_CLASS_LABELS_FOR_PROMPT=true
DO_PROP_SAMPLING_FOR_PROMPT=true
ICL_K=3

MODEL_ID=63794613/checkpoint-best # qwen1.5B full all (apr23)

# ************************************************************************
# removal test

# ROOM_TYPE=bedroom
# OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-with-qwen1.5b-all-removal-3/json
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --do-bedroom-testset --do-removal-test

ROOM_TYPE=livingroom
OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-with-qwen1.5b-all-removal-3/json
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --do-livingroom-testset --do-removal-test

# ROOM_TYPE=all
# OUTPUT_DIR_SCENES=./eval/samples/respace/instr/${ROOM_TYPE}-with-qwen1.5b-all-removal-3/json
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --do-removal-test

# ************************************************************************
# bedroom

# BON_LLM=2

# ROOM_TYPE=bedroom
# # OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-bon-${BON_LLM}/json
# OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json

# if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
#     ICL_FLAG="--do-icl-for-prompt"
# else
#     ICL_FLAG=""
# fi

# if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
#     CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
# else
#     CLASS_LABELS_FLAG=""
# fi

# if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
#     PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
# else
#     PROP_SAMPLING_FLAG=""
# fi

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678

# python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --do-bedroom-testset --resume

# ************************************************************************
# livingroom

# BON_LLM=8

# ROOM_TYPE=livingroom

# OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-bon-${BON_LLM}/json

# if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
#     ICL_FLAG="--do-icl-for-prompt"
# else
#     ICL_FLAG=""
# fi

# if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
#     CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
# else
#     CLASS_LABELS_FLAG=""
# fi

# if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
#     PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
# else
#     PROP_SAMPLING_FLAG=""
# fi

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678

# python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --do-livingroom-testset

# ************************************************************************
# all

# while true; do
#   sleep 600
# done

# BON_LLM=8

# ROOM_TYPE=all

# # OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-bon-${BON_LLM}/json
# OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json

# if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
#     ICL_FLAG="--do-icl-for-prompt"
# else
#     ICL_FLAG=""
# fi

# if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
#     CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
# else
#     CLASS_LABELS_FLAG=""
# fi

# if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
#     PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
# else
#     PROP_SAMPLING_FLAG=""
# fi

# # # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# # mkdir -p $OUTPUT_DIR_SCENES/1234
# # mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678

# python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="sherlock" --use-logfile --jid="${JOB_ID}" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --bon-llm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --seed-only=5678