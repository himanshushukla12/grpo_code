#!/bin/bash

# Script to start vLLM server on GPU 0 for GRPO training
# Run this first, then run train_single_gpu.sh in another terminal

set -e

#!/bin/bash

echo "=================================================="
echo "vLLM Server Startup - Ubuntu Dual GPU Setup"
echo "=================================================="
echo "GPU 0: vLLM Inference Server (this terminal)"
echo "GPU 1: Training Process (start in Terminal 2 after this)"
echo "=================================================="
echo ""
echo "⚠️  IMPORTANT: Keep this terminal open during training!"
echo "    Training process will connect to this server."
echo ""

# Check GPU 0 availability
echo "Checking GPU 0 availability..."
nvidia-smi -i 0 || {
    echo "ERROR: GPU 0 not available!"
    exit 1
}

# Set environment to use only GPU 0 for vLLM
export CUDA_VISIBLE_DEVICES=0

# Start vLLM server
echo "Launching vLLM server on GPU 0..."
echo "Server will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"

axolotl vllm-serve custom_dataset_config.yaml
