# ReSpace: training script for Stage 1 (SFT) on Qwen-2.5-1.5B model

# if using single GPU training, remove --multi_gpu flag

# set number of GPU tasks here:
export N_TASKS=2

# adjust these params if needed
export JOB_NUM_NODES=1
export NODEID=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# add --use-logfile if you want to log all stdout into a logfile
# add --multi-gpu directly after 'accelerate launch' if using > 1 GPU

# important: set this path to the model checkpoint you want to use from the SFT stage
# you need to provide the folder checkpoint-best (ideally) or checkpoint-last
# the models are saved under ./ckpts/
MODEL_PATH="<model_ckpt_id>/checkpoint-best"

# assume access to at least 2 GPUs, so we can use vLLM on one GPU and GRPO with gradients on the other
# if you want to use vLLM on the same GPU as GRPO, set N_TASKS=1 and remove --use-vllm. also change num_processes to 1

# GRPO
accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="$(uuidgen)" --run-id="apr23-qwen1.5Ball-grpo" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=".env" --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=6 --grpo-num-gen=6 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0 --grpo-temp=0.7