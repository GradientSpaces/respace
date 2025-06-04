#!/usr/bin/bash

#SBATCH --partition=serc
#SBATCH --nodes=1

#SBATCH --gpus=4

#SBATCH -C GPU_MEM:80GB

# #SBATCH --time=47:00:00
# #SBATCH --time=60:00:00
# #SBATCH --time=59:29:00
#SBATCH --time=47:59:00
# #SBATCH --time=11:59:00
# #SBATCH --time=05:29:00
# #SBATCH --time=23:59:00
# #SBATCH --time=9:59:00
# #SBATCH --time=57:59:00
# #SBATCH --time=00:39:00
# #SBATCH --time=01:40:00
# #SBATCH --time=0:59:00

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=96G
#SBATCH --job-name=sgllm
#SBATCH --output=logs/%j/out.log
#SBATCH --error=logs/%j/err.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export N_TASKS=4

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
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

ml load python/3.12.1
ml load cuda/12.4.0
ml load gcc/12.4.0
ml load system nvtop

# ml load python/3.9.0
# ml load cuda/12.1.1
# ml load gcc/10.3.0
# ml load system nvtop
# ml load py-pyopengl/3.1.5_py39

cd $HOME/stan-24-sgllm
git pull
source .venv_py312/bin/activate

pip install torch accelerate importlib_metadata
pip install --upgrade transformers

# pip uninstall trl -y
# pip install git+https://github.com/huggingface/trl.git
# pip install qwen-vl-utils #[decord]==0.0.8

# python -m pip install -r requirements.txt
# pip install wheel
# pip install flash-attn --no-build-isolation

# ************************************************************************
# interactive shell with a gpu

# while true; do
#   sleep 600
# done

# tmux
# export HF_HOME="${SCRATCH}/huggingface"
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export TOKENIZERS_PARALLELISM=false
# ml load python/3.12.1
# ml load cuda/12.4.0
# ml load gcc/12.4.0
# ml load system nvtop
# huggingface-cli login # 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

# cd $HOME/stan-24-sgllm
# git pull
# source ./.venv/bin/activate

# python src/train.py --use-cached-dataset --use-gpu --env=sherlock
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --test-jid="8c9377aa-8850-4eab-b370-8a8f12a619e6"

# ************************************************************************
#  stanley interactive
# ************************************************************************

# test
# xvfb-run -a accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --test-ckpt="57268357/checkpoint-best" --run-id="dec19-r256-rotv3-1B-fix-v2" --use-cached-dataset --use-gpu --env=stanley

# train
# xvfb-run -a accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="feb11-test" --use-cached-dataset --use-gpu --env=stanley --test-bs=4 --do-sanity-check --room-type="bedroom" --dvc-batch-size=8 --jid="$(uuidgen)"

# grpo
# xvfb-run -a accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --test-ckpt="59985207/checkpoint-best" --run-id="feb11-test" --use-cached-dataset --use-gpu --env=stanley --test-bs=4 --do-sanity-check --room-type="bedroom" --dvc-batch-size=2 --do-grpo --jid="$(uuidgen)"

# ************************************************************************
# train (sherlock)
# ************************************************************************

# sanity (only do 1-2 epochs for 1h)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar10-llama1B-sanity" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=32 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar10-qwen0.5B-sanity" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=24 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0

# SFT (lora)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar11-llama1B" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=32 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar11-qwen0.5B" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=24 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar11-qwen1.5B" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=16 --llm="qwen-2.5-1.5B" --dvc-batch-size=4 --gas-steps=8 --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="mar13-llama3B-2gpu" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --epochs=100 --test-bs=16 --llm="llama-3.2-3B" --dvc-batch-size=4 --gas-steps=8 --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0

# SFT (full / bedroom)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen0.5B-full-bdrm" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-llama1B-full-bdrm" --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen1.5B-full-bdrm" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-1.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8

# SFT (full / livingroom)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen0.5B-full-lvngrm" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-0.5B" --room-type="livingroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-llama1B-full-lvngrm" --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="llama-3.2-1B" --room-type="livingroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen1.5B-full-lvngrm" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-1.5B" --room-type="livingroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8

# SFT (full / all)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen0.5B-full-all" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-0.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-llama1B-full-all" --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="llama-3.2-1B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr23-qwen1.5B-full-all" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8

# GRPO
# accelerate launch --multi_gpu --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr29-llama1Bfull-grpo-v5-5e-5-g8-test" --test-ckpt="63772118/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=8 --grpo-num-gen=8 --gas-steps=8 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5"
# MODEL_PATH="63772118/checkpoint-best" # llama1B/bedrooms
MODEL_PATH="63794613/checkpoint-best" # qwen1.5B/all
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=2
# trl vllm-serve --model "./ckpts/${MODEL_PATH}" --model-impl=transformers --max-model-len=3000 --gpu-memory-utilization 0.8 --port 8000 > "./logs/${JOB_ID}/vllm.log" 2>&1 &saf
# export VLLM_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# trl vllm-serve --host $MASTER_ADDR --model "./ckpts/${MODEL_PATH}" --gpu-memory-utilization 0.8 --port $VLLM_PORT > "./logs/${JOB_ID}/vllm.log" 2>&1 &
# VLLM_PID=$!
# sleep 30
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="apr29-llama1Bfull-grpo-v5-1e-5-g8-test" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=8 --grpo-num-gen=8 --gas-steps=8 --grpo-learning-rate=1e-6 --use-vllm --grpo-reward-version="v5"
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may02-qwen1.5Ball-grpo-v5-5e-6-g4" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-6 --use-vllm --grpo-reward-version="v5" --vllm-host="${MASTER_ADDR}" --vllm-port="${VLLM_PORT}"
# kill $VLLM_PID

# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-5e-6-g4" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-6 --use-vllm --grpo-reward-version="v5"
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-1e-6-g4" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=1e-6 --use-vllm --grpo-reward-version="v5"
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-0e-6-g4" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=0e-6 --use-vllm --grpo-reward-version="v5"
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-1e-5-g4" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=1e-5 --use-vllm --grpo-reward-version="v5"
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-5e-5-g4-ntrl" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-5e-5-g4-ntrl-beta0.2" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.2
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may08-qwen1.5Ball-grpo-v5-1e-4-g4-ntrl-beta0.0" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=1e-4 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may04-qwen1.5Ball-grpo-v5-5e-4-g4-ntrl-beta0.0" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-4 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0
accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may11-qwen1.5Ball-grpo-v5-5e-5-g6-ntrl-beta0.0-temp0.7" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=6 --grpo-num-gen=6 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0 --grpo-temp=0.7
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may10-qwen1.5Ball-grpo-v5-5e-5-g4-ntrl-beta0.0-temp0.9" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0 --grpo-temp=0.9
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="may10-qwen1.5Ball-grpo-v5-1e-4-g4-ntrl-beta0.0-temp0.9" --test-ckpt=$MODEL_PATH --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-1.5B" --room-type="all" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=4 --grpo-num-gen=4 --gas-steps=16 --grpo-learning-rate=1e-4 --use-vllm --grpo-reward-version="v5" --grpo-do-neutral-rewards --grpo-beta-kl=0.0 --grpo-temp=0.9

# DPO
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="mar15-qwen0.5full-dpo-5e-7" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8 --dpo-learning-rate=5e-7 --use-vllm
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="mar15-qwen0.5full-dpo-5e-6" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8 --dpo-learning-rate=5e-6 --use-vllm
# accelerate launch --debug --num_processes="$((N_TASKS-1))" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="mar15-qwen0.5full-dpo-5e-5" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8 --dpo-learning-rate=5e-5 --use-vllm

# ************************************************************************
# debug
# --jid="$(uuidgen)"

#accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/main.py --run-id="mar15-qwen0.5full-grpo-5e-5" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-grpo --dvc-batch-size=8 --grpo-num-gen=8 --gas-steps=16 --grpo-learning-rate=5e-5 --use-vllm --jid="$(uuidgen)"
#accelerate launch --debug --num_processes=2 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/main.py --run-id="mar15-qwen0.5full-dpo-5e-5" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8 --dpo-learning-rate=5e-5 --jid="$(uuidgen)" --use-vllm

#accelerate launch --multi_gpu --debug --num_processes=2 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/main.py --run-id="mar15-qwen0.5full-dpo" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8
#accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/main.py --run-id="mar15-qwen0.5full-dpo" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=4 --gas-steps=8 --jid="$(uuidgen)" --do-sanity-check --use-vllm

# SFT / sherlock
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="mar06-qwen0.5B-resume" --use-cached-dataset --use-gpu --env=sherlock --lora-rank=8 --epochs=100 --test-bs=24 --llm="qwen-2.5-0.5B" --room-type="bedroom" --do-augm --lambda-instr-exp=0.0 --do-sanity-check --jid="$(uuidgen)"

# SFT / stanley
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="mar06-qwen0.5B-resume" --use-cached-dataset --use-gpu --env=stanley --lora-rank=8 --epochs=100 --test-bs=24 --llm="qwen-2.5-0.5B" --room-type="bedroom" --do-augm --lambda-instr-exp=0.0 --do-sanity-check --jid="$(uuidgen)" --dvc-batch-size=4

# GRPO / sherlock
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --test-ckpt="59985207/checkpoint-best" --run-id="feb11-test" --use-cached-dataset --use-gpu --env=sherlock --test-bs=4 --do-sanity-check --room-type="bedroom" --dvc-batch-size=2 --grpo-num-gen=2 --do-grpo --jid="$(uuidgen)"
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/main.py --test-ckpt="61414298/checkpoint-best" --run-id="mar18-dpo" --use-cached-dataset --use-gpu --env=sherlock --epochs=100 --test-bs=4 --llm="qwen-2.5-0.5B" --room-type="bedroom" --dvc-batch-size=2 --grpo-num-gen=2 --do-grpo --dvc-batch-size=8 --grpo-num-gen=8 --gas-steps=64 --jid="$(uuidgen)" --do-sanity-check --use-vllm

# GRPO / stanley
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --test-ckpt="59985207/checkpoint-best" --run-id="feb11-test" --use-cached-dataset --use-gpu --env=stanley --test-bs=4 --do-sanity-check --room-type="bedroom" --dvc-batch-size=2 --grpo-num-gen=2 --do-grpo --jid="$(uuidgen)"

# SFT / macbook
# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision bf16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="mar08-llama1B-grpo" --use-cached-dataset --env=local --lora-rank=8 --epochs=100 --test-bs=24 --llm="llama-3.2-1B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-sanity-check --jid="$(uuidgen)" --dvc-batch-size=4 --test-ckpt="60083268/checkpoint-best" --do-grpo --grpo-num-gen=6

# DPO / stanley
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="${JOB_ID}" --run-id="mar15-qwen0.5full-dpo" --test-ckpt="61414298/checkpoint-best" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --do-dpo --dvc-batch-size=1 --gas-steps=32

# ************************************************************************
# random (stanley)

# xvfb-run -a accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision fp16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="feb04-r256-1B" --env=stanley --use-cached-dataset --room-type="bedroom" --prep-data-only --jid="$(uuidgen)"

# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision fp16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="feb18-qw-0.5B" --use-cached-dataset --use-gpu --env=stanley --lora-rank=256 --epochs=100 --test-bs=32 --llm="qwen-2.5-0.5B" --room-type="bedroom" --do-augm --lambda-instr-exp=0.0 --jid="$(uuidgen)" --do-sanity-check --use-wandb

# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision fp16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="mar06-grpo" --test-ckpt="59985207/checkpoint-best" --use-cached-dataset --use-gpu --env=stanley --lora-rank=256 --epochs=100 --test-bs=32 --llm="llama-3.2-1B" --room-type="bedroom" --do-augm --lambda-instr-exp=0.0 --jid="$(uuidgen)" --do-grpo --test-ckpt="59985207/checkpoint-best" --do-sanity-check --dvc-batch-size=2 --grpo-num-gen=2

# accelerate launch --debug --num_processes=1 --num_machines=1 --mixed_precision fp16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" --dynamo_backend=no src/train.py --test-jid="56902819/checkpoint-best" --run-id="dec11-r256-rotv3-1B" --use-cached-dataset --env=stanley --do-run-interactive

# accelerate launch --multi_gpu --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_preci
# accelerate launch --multi_gpu --debug --num_processes=2 --num_machines=1 --mixed_precision fp16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="nov06-r8-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --do-augm-rot --test-bs=4 --multi-gpu --use-wandb

# accelerate launch --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision fp16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="nov07-r8-norot-acc1gpu" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --test-bs=4 --multi-gpu --use-wandb
# accelerate launch --multi_gpu --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision fp16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="nov07-r8-norot-acc2gpus" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --test-bs=4 --multi-gpu --use-wandb

# $HOME/stan-24-sgllm/.venv/bin/python src/train.py --jid="${JOB_ID}" --run-id="oct29-r8-norot" --use-cached-dataset --use-gpu --use-logfile --use-wandb --env=sherlock --lora-rank=8
# $HOME/stan-24-sgllm/.venv/bin/python src/train.py --jid="${JOB_ID}" --run-id="oct27-r8-rot" --use-cached-dataset --use-gpu --use-logfile --use-wandb --env=sherlock --lora-rank=8 --do-augm-rot

# python src/train.py --jid="${JOB_ID}" --run-id="oct29-r256-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=256 --do-augm-rot --do-sanity-check --use-wandb
# python src/train.py --jid="${JOB_ID}" --run-id="oct31-r16-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=16 --do-augm-rot --do-sanity-check --use-wandb --test-bs=4

# accelerate launch --multi_gpu --num_processes $SLURM_NTASKS --num_machines $SLURM_JOB_NUM_NODES src/train.py --jid="${JOB_ID}" --run-id="oct29-r8-rot-acc" --use-cached-dataset --use-gpu --use-logfile --use-wandb --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check
# accelerate launch --config_file src/scripts/accelerate_config_resolved.yaml src/train.py --jid="${JOB_ID}" --run-id="oct29-r8-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check --test-bs=16 --use-wandb
# accelerate launch --multi_gpu --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision fp16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="oct29-r8-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check --test-bs=4 --multi-gpu --use-wandb
# accelerate launch --multi_gpu --num_processes="2" --num_machines="1" --mixed_precision fp16 --machine_rank="0" --main_process_ip="localhost" --main_process_port="29500" --dynamo_backend=no src/train.py --run-id="oct29-r8-rot" --use-cached-dataset --use-gpu --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check --test-bs=4 --multi-gpu --use-wandb
# accelerate launch --multi_gpu --num_processes="2" --num_machines="1" --mixed_precision fp16 --machine_rank="0" --main_process_ip="localhost" --main_process_port="29500" --dynamo_backend=no src/train.py --jid="$(uuidgen)" --run-id="oct29-r8-rot" --use-cached-dataset --use-gpu --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check --test-bs=2 --multi-gpu

# accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision fp16 --machine_rank=0 --main_process_ip="127.0.0.1" --main_process_port="29500" src/train.py --run-id="oct29-r8-rot" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --do-augm-rot --do-sanity-check --test-bs=4 --multi-gpu
# accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --machine_rank 0 --main_process_ip "127.0.0.1" --main_process_port "1234" --dynamo_backend=no src/train.py --run-id="nov07-r8-norot-acc1gpu" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --test-bs=4 --multi-gpu --do-augm-rot --do-sanity-check --use-wandb
# accelerate launch --debug --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --machine_rank 0 --main_process_ip "127.0.0.1" --main_process_port "1234" --dynamo_backend=no src/train.py --run-id="nov08-r8-norot-acc2gpu" --jid="$(uuidgen)" --use-cached-dataset --use-gpu --env=sherlock --lora-rank=8 --multi-gpu --epochs=2 --test-bs=4 --do-augm-rot --do-sanity-check
# accelerate launch --debug --num_processes="${N_TASKS}" --num_machines="${SLURM_JOB_NUM_NODES}" --mixed_precision fp16 --machine_rank="${SLURM_NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/train.py --jid="${JOB_ID}" --run-id="nov09-r8-rotv2" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=8 --epochs=20 --test-bs=16 --multi-gpu --do-augm-rot --use-wandb

# accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --machine_rank 0 --main_process_ip "127.0.0.1" --main_process_port "1234" --dynamo_backend=no src/train.py --test-jid="55340446" --use-cached-dataset --use-gpu --env=sherlock --test-bs=4 --multi-gpu --do-sanity-check
# accelerate launch --debug --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --machine_rank 0 --main_process_ip "127.0.0.1" --main_process_port "1234" --dynamo_backend=no src/train.py --test-jid="55340446" --use-cached-dataset --use-gpu --env=sherlock --test-bs=4 --multi-gpu --do-sanity-check

# $HOME/stan-24-sgllm/.venv/bin/python src/train.py --jid="${JOB_ID}" --use-cached-dataset --use-gpu --use-logfile --env=sherlock --lora-rank=64 --run-id="first-exp-lora-r64"
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --lora-rank=256 --run-id="oct-24-r2" --do-augm-rot --do-sanity-check
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --lora-rank=16 --run-id="oct-24-r2" --do-augm-rot --do-sanity-check
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --run-id="oct-29-r2" --test-jid="55011685" --do-dpo

# ************************************************************************
#  test and localhost
# ************************************************************************

# accelerate launch --debug --num_processes="1" --num_machines="1" --machine_rank="0" --main_process_ip="localhost" --main_process_port="29500" --dynamo_backend=no src/train.py --jid="$(uuidgen)" --run-id="nov11-r8-rotv3" --use-cached-dataset --env=local --lora-rank=8 --epochs=10 --llama="1B" --do-augm-rot --test-bs=2 --do-sanity-check

# $HOME/stan-24-sgllm/.venv/bin/python src/train.py --jid="${JOB_ID}" --use-cached-dataset --use-gpu --use-logfile --test-jid="53004368"
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --test-jid="8c9377aa-8850-4eab-b370-8a8f12a619e6"

# python src/train.py --use-cached-dataset --use-gpu --env=sherlock --test-jid="47807bea-83fb-4c2c-9466-5bab111e3ee2"
# python src/train.py --use-cached-dataset --use-gpu --env=sherlock

# ************************************************************************

# python src/train.py --use-gpu --lora-rank=8 --run-id="some-exp-r8"

# TODO for multi GPU run:
# python -m torch.distributed.launch src/train.py

# SUBMIT JOB
# sbatch src/scripts/run_sherlock.sh

# KILL JOB
# scancel <job_id>

# check how busy
# sh_part
# squeue -p serc

# check job:
# squeue -u $USER
# sjobs

# check disk space
# sh_quota

# INTERACTIVE SHELL
# salloc -p serc --gpus 1
