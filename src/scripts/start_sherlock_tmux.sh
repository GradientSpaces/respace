#!/bin/bash

# Check if an ID argument is passed
if [ -z "$1" ]; then
  echo "Usage: $0 <server_id>"
  exit 1
fi

SERVER_ID="$1"

# Define the tmux session name
TMUX_SESSION="martin"

# SSH and start tmux session with all the setup commands
ssh -t "$SERVER_ID" << EOF
  # Start or attach to tmux session
  tmux new-session -d -s $TMUX_SESSION || tmux attach -t $TMUX_SESSION
  
  # Run initialization commands inside tmux session
  tmux send-keys -t $TMUX_SESSION "
    export HF_HOME=\"\${SCRATCH}/huggingface\"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export TOKENIZERS_PARALLELISM=false
    ml load python/3.12.1
    ml load cuda/12.4.0
    ml load gcc/12.4.0
    ml load system nvtop
    huggingface-cli login --token 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
    cd \$HOME/stan-24-sgllm
    git pull
    source .venv_py312/bin/activate
  " C-m
EOF
