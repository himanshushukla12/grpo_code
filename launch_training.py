#!/usr/bin/env python3
"""
Training launcher for dual GPU setup on Ubuntu.
GPU 0: vLLM server, GPU 1: Training
"""

import os
import sys
import torch
import subprocess
import requests
import time
from pathlib import Path

def check_gpu_setup():
    """Check dual GPU availability and setup."""
    print("=== GPU Setup Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    
    if device_count < 2:
        print("WARNING: Less than 2 GPUs available. This setup expects:")
        print("  - GPU 0: vLLM server")
        print("  - GPU 1: Training")
        print(f"Found only {device_count} GPU(s)")
        
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return device_count >= 1  # At least 1 GPU needed

def check_vllm_server():
    """Check if vLLM server is running and accessible."""
    print("=== vLLM Server Check ===")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ vLLM server is running and accessible")
            return True
        else:
            print(f"✗ vLLM server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ vLLM server not accessible: {e}")
        print("\nTo start vLLM server, run in another terminal:")
        print("  chmod +x start_vllm_server.sh")
        print("  ./start_vllm_server.sh")
        print("\nOr manually:")
        print("  CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve custom_dataset_config.yaml")
        return False

def setup_training_environment():
    """Set up environment variables for training on GPU 1."""
    print("=== Training Environment Setup ===")
    
    # Set environment for training on GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["MAX_WORKERS"] = "64"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    # Clear any existing accelerate config that might cause issues
    if "ACCELERATE_CONFIG_FILE" in os.environ:
        del os.environ["ACCELERATE_CONFIG_FILE"]
    
    print("Environment variables set:")
    for key in ["CUDA_VISIBLE_DEVICES", "MAX_WORKERS", "WORLD_SIZE", "RANK", "LOCAL_RANK"]:
        print(f"  {key}={os.environ.get(key)}")

def clear_accelerate_cache():
    """Clear accelerate cache to avoid configuration conflicts."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    if cache_dir.exists():
        print(f"Clearing accelerate cache: {cache_dir}")
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

def run_training():
    """Run the training with proper configuration."""
    print("=== Starting Training ===")
    
    config_file = Path("custom_dataset_config.yaml")
    accelerate_config = Path("accelerate_config.yaml")
    
    if not config_file.exists():
        print(f"ERROR: {config_file} not found!")
        return False
    
    if not accelerate_config.exists():
        print(f"ERROR: {accelerate_config} not found!")
        return False
    
    cmd = [
        "accelerate", "launch",
        "--config_file", str(accelerate_config),
        "--num_processes", "1",
        "--num_machines", "1", 
        "--machine_rank", "0",
        "--main_process_port", "29500",
        "-m", "axolotl.cli.train",
        str(config_file)
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nTraining will use GPU 1 while vLLM uses GPU 0")
    print("Press Ctrl+C to stop training\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def main():
    print("=== Dual GPU GRPO Training Launcher (Ubuntu) ===")
    print("GPU 0: vLLM Server | GPU 1: Training")
    print("=" * 50)
    
    # Check GPU setup
    if not check_gpu_setup():
        print("GPU setup check failed. Exiting.")
        sys.exit(1)
    
    # Check vLLM server
    if not check_vllm_server():
        print("vLLM server check failed. Please start the server first.")
        sys.exit(1)
    
    # Setup training environment
    setup_training_environment()
    
    # Clear accelerate cache
    clear_accelerate_cache()
    
    # Initialize CUDA context on training GPU
    try:
        # This will use GPU 1 due to CUDA_VISIBLE_DEVICES=1
        torch.cuda.init()
        device = torch.cuda.current_device()
        print(f"CUDA initialized successfully. Training will use device {device}")
    except Exception as e:
        print(f"Failed to initialize CUDA: {e}")
        sys.exit(1)
    
    # Run training
    success = run_training()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
