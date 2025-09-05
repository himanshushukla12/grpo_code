#!/bin/bash

echo "=============================================="
echo "GRPO Training Script - Ubuntu Dual GPU Setup"
echo "=============================================="
echo "GPU 0: vLLM Server (Terminal 1 - must be running)"
echo "GPU 1: Training Process (Terminal 2 - this script)"
echo "=============================================="

# Check if vLLM server is running in Terminal 1
echo "Checking vLLM server connectivity..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ vLLM server is accessible (Terminal 1 running correctly)"
else
    echo "✗ vLLM server not accessible!"
    echo ""
    echo "Please start vLLM server in Terminal 1 first:"
    echo "  chmod +x start_vllm_server.sh"
    echo "  ./start_vllm_server.sh"
    echo ""
    echo "Or manually:"
    echo "  CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve custom_dataset_config.yaml"
    echo ""
    echo "Then run this training script in Terminal 2."
    exit 1
fi

# Dual GPU GRPO training on Ubuntu
# GPU 0: vLLM server, GPU 1: Training
# This script handles CUDA device initialization properly

set -e

echo "Starting GRPO training with dual GPU setup (GPU 0: vLLM, GPU 1: Training)..."

# Check GPU availability
echo "Checking available GPUs..."
nvidia-smi --list-gpus

# Set environment variables for training on GPU 1
export CUDA_VISIBLE_DEVICES=1  # Training will use GPU 1
export MAX_WORKERS=64
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false  # Fix tokenizer warnings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization

# Ensure CUDA is properly initialized
echo "Testing CUDA on training GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

# Clear any problematic accelerate cache
echo "Clearing accelerate cache..."
rm -rf ~/.cache/huggingface/accelerate/

# Double-check vLLM server is still running
echo "Final vLLM connectivity check..."
curl -f http://localhost:8000/health || {
    echo "ERROR: vLLM server not responding!"
    echo "Please ensure vLLM server is running in Terminal 1"
    exit 1
}

# Run training with explicit accelerate config
echo "Starting training on GPU 1..."
echo "Using accelerate to avoid DTensor issues..."
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    -m axolotl.cli.train custom_dataset_config.yaml

echo "Training completed!"
