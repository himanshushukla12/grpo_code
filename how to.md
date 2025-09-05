# How to Train GRPO Models: Complete GPU Configuration Guide

This guide covers all possible GPU configurations for GRPO (Group Relative Policy Optimization) training using Axolotl and vLLM.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Custom Dataset Preparation](#custom-dataset-preparation)
- [Single GPU Setup](#single-gpu-setup)
- [Dual GPU Setup (Recommended)](#dual-gpu-setup-recommended)
- [Multi-GPU Setup (3+ GPUs)](#multi-gpu-setup-3-gpus)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)
- [Configuration Reference](#configuration-reference)

## Overview

GRPO training requires two components running simultaneously:
1. **vLLM Inference Server**: Generates code completions during training
2. **Training Process**: Runs the actual GRPO reinforcement learning

The key is to allocate GPU resources efficiently between these two processes.

## Prerequisites

### Required Software
```bash
# Install Axolotl with vLLM support
pip install axolotl[vllm,flash-attn]==0.8.0

# Verify CUDA installation
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Project Structure
```
grpo_code/
├── custom_dataset_config.yaml    # Main training configuration
├── finetuning_dataset.jsonl      # Your instruction-response dataset
├── grpo_code/                     # Custom transforms and rewards
├── start_vllm_server.sh          # vLLM server startup script
├── train_single_gpu.sh           # Training script for single/dual GPU
└── train_multi_gpu.sh            # Training script for multi-GPU
```

## Custom Dataset Preparation

GRPO training requires a properly formatted dataset with instruction-response pairs for code generation tasks.

### Dataset Format

Your dataset should be in JSONL format with each line containing:

```json
{
  "instruction": "Write a Python function to calculate the factorial of a number",
  "response": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
}
```

### Supported Dataset Formats

#### 1. Instruction-Response Format (Recommended)
```json
{"instruction": "Create a function to sort a list", "response": "def sort_list(lst):\n    return sorted(lst)"}
{"instruction": "Write a class for a simple calculator", "response": "class Calculator:\n    def add(self, a, b):\n        return a + b"}
```

#### 2. Input-Output Format
```json
{"input": "Write a function to reverse a string", "output": "def reverse_string(s):\n    return s[::-1]"}
```

#### 3. Custom Format with Transform
```json
{"problem": "Find the maximum in a list", "solution": "def find_max(lst):\n    return max(lst)", "difficulty": "easy"}
```

### Dataset Transforms

The project includes custom transforms to convert different formats to GRPO-compatible format:

#### Using Built-in Transforms

1. **Basic Instruction-Response Transform**:
```yaml
# In your config file
datasets:
  - path: ./your_dataset.jsonl
    type: grpo_code.instruction_response_transform
```

2. **Enhanced Transform with Tests**:
```yaml
datasets:
  - path: ./your_dataset.jsonl
    type: grpo_code.instruction_response_with_tests_transform
```

#### Creating Custom Transforms

If your dataset has a unique format, create a custom transform in `grpo_code/custom_transforms.py`:

```python
def my_custom_transform(sample):
    """Transform your custom format to GRPO format."""
    return {
        "instruction": sample["your_instruction_field"],
        "response": sample["your_response_field"]
    }
```

Then register it in `grpo_code/__init__.py`:
```python
from .custom_transforms import my_custom_transform
```

### Dataset Quality Guidelines

#### 1. Code Quality
- **Syntax**: All code should be syntactically correct
- **Functionality**: Code should actually solve the given problem
- **Style**: Follow consistent coding style (PEP8 for Python)

#### 2. Instruction Clarity
- **Specific**: Clear, unambiguous problem statements
- **Complete**: Include all necessary context
- **Varied**: Mix of different problem types and difficulties

#### 3. Response Quality
- **Correct**: Solutions should work correctly
- **Efficient**: Prefer efficient algorithms when possible
- **Readable**: Well-commented and structured code

### Dataset Validation

Test your dataset before training:

```bash
# Test transforms with sample data
python simple_test_transforms.py

# Validate dataset format
python -c "
import json
with open('your_dataset.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            assert 'instruction' in data and 'response' in data
            print(f'✓ Line {i+1}: Valid')
            if i >= 4:  # Check first 5 lines
                break
        except Exception as e:
            print(f'✗ Line {i+1}: {e}')
"
```

### Dataset Size Recommendations

| Dataset Size | Training Time | GPU Memory | Use Case |
|--------------|---------------|------------|----------|
| 1K-5K samples | 1-3 hours | 16GB+ | Testing/Prototyping |
| 10K-50K samples | 6-24 hours | 24GB+ | Small-scale training |
| 50K-100K samples | 1-3 days | 40GB+ | Production training |
| 100K+ samples | 3+ days | 48GB+ | Large-scale training |

### Example Dataset Creation

Here's how to create a dataset from scratch:

```python
import json

def create_training_dataset():
    """Create a sample GRPO training dataset."""
    samples = [
        {
            "instruction": "Write a function to check if a number is prime",
            "response": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "instruction": "Create a function to find the GCD of two numbers",
            "response": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"
        },
        {
            "instruction": "Write a function to generate Fibonacci sequence",
            "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        }
    ]
    
    with open('custom_dataset.jsonl', 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

# Create the dataset
create_training_dataset()
```

### Configuration for Custom Datasets

Update your configuration file to use your custom dataset:

```yaml
# Custom dataset configuration
datasets:
  - path: ./your_custom_dataset.jsonl
    type: grpo_code.instruction_response_transform

# Optional: Dataset processing settings
dataset_prepared_path: ./prepared_data
dataset_processes: 1
skip_prepare_dataset: false
val_set_size: 0.1  # Use 10% for validation

# Reward functions for code evaluation
trl:
  reward_funcs:
    - grpo_code.soft_format_reward_func  # Syntax and format checking
    - grpo_code.code_execution_reward_func  # Execution and correctness
```

### Common Dataset Issues and Solutions

#### Issue 1: JSON Parsing Errors
```bash
# Check for malformed JSON
python -m json.tool your_dataset.jsonl
```

#### Issue 2: Empty or Missing Fields
```python
# Validate required fields
import json
with open('dataset.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if not data.get('instruction') or not data.get('response'):
            print(f"Line {i+1}: Missing required fields")
```

#### Issue 3: Code Syntax Errors
```python
# Validate Python syntax
import ast
for sample in dataset:
    try:
        ast.parse(sample['response'])
    except SyntaxError as e:
        print(f"Syntax error in: {sample['instruction']}")
```

### Advanced Dataset Features

#### 1. Multi-turn Conversations
```json
{
  "instruction": "Create a class for managing a bank account",
  "response": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    \n    def deposit(self, amount):\n        self.balance += amount\n        return self.balance",
  "follow_up": "Add a withdraw method",
  "follow_up_response": "def withdraw(self, amount):\n    if amount <= self.balance:\n        self.balance -= amount\n        return self.balance\n    else:\n        raise ValueError('Insufficient funds')"
}
```

#### 2. Code with Tests
```json
{
  "instruction": "Write a function to calculate the area of a circle",
  "response": "import math\n\ndef circle_area(radius):\n    return math.pi * radius ** 2",
  "test_cases": [
    "assert abs(circle_area(1) - 3.14159) < 0.001",
    "assert circle_area(0) == 0",
    "assert circle_area(2) > circle_area(1)"
  ]
}
```

#### 3. Difficulty Levels
```json
{
  "instruction": "Implement binary search",
  "response": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
  "difficulty": "medium",
  "tags": ["algorithms", "search", "arrays"]
}
```

## Single GPU Setup

**Use Case**: Limited hardware, shared GPU memory between vLLM and training.

### Memory Requirements
- **Minimum**: 24GB VRAM
- **Recommended**: 40GB+ VRAM

### Configuration

1. **Create single GPU config** (`single_gpu_config.yaml`):
```yaml
base_model: Qwen/Qwen2.5-3B-Instruct

# vLLM configuration - shared GPU
vllm:
    host: 0.0.0.0
    port: 8000
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.6  # Conservative for shared GPU
    dtype: auto

rl: grpo
trl:
  beta: 0.001
  use_vllm: true
  vllm_server_host: 0.0.0.0
  vllm_server_port: 8000
  num_generations: 2  # Reduced for memory
  max_completion_length: 256

# Memory-optimized training settings
gradient_accumulation_steps: 16
micro_batch_size: 2
sequence_len: 512
```

2. **Terminal 1 - Start vLLM Server**:
```bash
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve single_gpu_config.yaml
```

3. **Terminal 2 - Start Training** (after vLLM is running):
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file accelerate_config.yaml \
  --num_processes 1 --num_machines 1 --machine_rank 0 \
  -m axolotl.cli.train single_gpu_config.yaml
```

### Single GPU Script
Create `train_single_gpu_shared.sh`:
```bash
#!/bin/bash
echo "Single GPU Shared Memory Training"
echo "GPU 0: vLLM + Training (shared)"

# Check vLLM server
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "ERROR: Start vLLM server first in Terminal 1"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file accelerate_config.yaml \
  --num_processes 1 -m axolotl.cli.train single_gpu_config.yaml
```

## Dual GPU Setup (Recommended)

**Use Case**: Optimal performance with dedicated GPU for each component.

### Memory Requirements
- **vLLM GPU**: 16GB+ VRAM
- **Training GPU**: 24GB+ VRAM

### Step-by-Step Setup

#### 1. Terminal 1 - vLLM Server (GPU 0)
```bash
# Start vLLM server and keep it running
chmod +x start_vllm_server.sh
./start_vllm_server.sh

# Or manually:
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve custom_dataset_config.yaml
```

#### 2. Terminal 2 - Training Process (GPU 1)
```bash
# Start training (connects to vLLM in Terminal 1)
chmod +x train_single_gpu.sh
./train_single_gpu.sh

# Or manually:
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file accelerate_config.yaml \
  --num_processes 1 --num_machines 1 --machine_rank 0 \
  -m axolotl.cli.train custom_dataset_config.yaml
```

#### 3. Configuration (`custom_dataset_config.yaml`)
```yaml
base_model: Qwen/Qwen2.5-3B-Instruct

# vLLM configuration for dedicated GPU 0
vllm:
    host: 0.0.0.0
    port: 8000
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.8  # Aggressive for dedicated GPU
    dtype: auto

rl: grpo
trl:
  beta: 0.001
  use_vllm: true
  vllm_server_host: 0.0.0.0
  vllm_server_port: 8000
  num_generations: 4
  max_completion_length: 256

# Optimized for dedicated training GPU
gradient_accumulation_steps: 8
micro_batch_size: 4
sequence_len: 512
```

## Multi-GPU Setup (3+ GPUs)

**Use Case**: Maximum performance, multiple vLLM servers or distributed training.

### Configuration Options

#### Option A: Multiple vLLM Servers + Dedicated Training GPUs
```
GPU 0-1: vLLM Server (tensor parallel)
GPU 2-3: Training Process (data parallel)
```

#### Option B: Load Balancing
```
GPU 0: vLLM Server 1 (port 8000)
GPU 1: vLLM Server 2 (port 8001)
GPU 2-3: Training Process
```

### Multi-GPU Training Script
Create `train_multi_gpu.sh`:
```bash
#!/bin/bash
echo "Multi-GPU GRPO Training Setup"
echo "GPUs 0-1: vLLM Servers"
echo "GPUs 2-3: Training Process"

# Check multiple vLLM servers
check_vllm_server() {
    local port=$1
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "✓ vLLM server on port $port is accessible"
        return 0
    else
        echo "✗ vLLM server on port $port not accessible"
        return 1
    fi
}

# Verify all vLLM servers are running
if ! check_vllm_server 8000 || ! check_vllm_server 8001; then
    echo "ERROR: Start all vLLM servers first"
    echo "Terminal 1: CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve config_gpu0.yaml"
    echo "Terminal 2: CUDA_VISIBLE_DEVICES=1 axolotl vllm-serve config_gpu1.yaml"
    exit 1
fi

# Training on GPUs 2-3
export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Multi-GPU training with data parallelism
accelerate launch --config_file accelerate_multi_gpu.yaml \
  --num_processes 2 --num_machines 1 --machine_rank 0 \
  -m axolotl.cli.train multi_gpu_config.yaml
```

### Multi-GPU Configuration
Create `multi_gpu_config.yaml`:
```yaml
base_model: Qwen/Qwen2.5-3B-Instruct

# Multiple vLLM servers for load balancing
vllm:
    host: 0.0.0.0
    port: 8000  # Primary server
    tensor_parallel_size: 2  # GPUs 0-1
    gpu_memory_utilization: 0.8

rl: grpo
trl:
  beta: 0.001
  use_vllm: true
  vllm_server_host: 0.0.0.0
  vllm_server_port: 8000
  num_generations: 8
  max_completion_length: 512

# Multi-GPU training settings
gradient_accumulation_steps: 4
micro_batch_size: 8
sequence_len: 1024

# Enable distributed data parallel
ddp_find_unused_parameters: false
```

### Accelerate Config for Multi-GPU
Create `accelerate_multi_gpu.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2  # Number of training GPUs
rdzv_backend: static
same_network: true
use_cpu: false
```

## Memory Optimization

### For Memory-Constrained Setups

1. **Reduce Model Size**:
```yaml
load_in_8bit: true  # Enable 8-bit quantization
# OR
load_in_4bit: true  # Enable 4-bit quantization
```

2. **Optimize Batch Sizes**:
```yaml
# Smaller batches, more accumulation
micro_batch_size: 2
gradient_accumulation_steps: 16

# Shorter sequences
sequence_len: 256
max_completion_length: 128
```

3. **Gradient Checkpointing**:
```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

4. **Environment Variables**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
torch.OutOfMemoryError: CUDA out of memory
```
**Solutions**:
- Reduce `micro_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `sequence_len`
- Enable quantization (`load_in_8bit: true`)

#### 2. vLLM Server Not Accessible
```
✗ vLLM server not accessible
```
**Solutions**:
```bash
# Check if server is running
curl http://localhost:8000/health

# Restart vLLM server
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve config.yaml

# Check port conflicts
netstat -tlnp | grep 8000
```

#### 3. DTensor Mixed with Tensor Error
```
RuntimeError: aten.cat.default: got mixed torch.Tensor and DTensor
```
**Solutions**:
- Use accelerate launcher instead of direct python
- Set `distributed_type: 'NO'` in accelerate config
- Disable torch compile: `torch_compile: false`

#### 4. Master Address Error
```
ValueError: Error initializing torch.distributed using env://
```
**Solutions**:
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
```

### Debugging Commands

```bash
# Check GPU usage
nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Test vLLM connectivity
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "prompt": "def hello():", "max_tokens": 50}'
```

## Configuration Reference

### Essential Parameters

| Parameter | Single GPU | Dual GPU | Multi-GPU | Description |
|-----------|------------|----------|-----------|-------------|
| `micro_batch_size` | 2 | 4 | 8 | Samples per forward pass |
| `gradient_accumulation_steps` | 16 | 8 | 4 | Steps before optimizer update |
| `num_generations` | 2 | 4 | 8 | Completions per prompt |
| `sequence_len` | 512 | 512 | 1024 | Maximum sequence length |
| `gpu_memory_utilization` | 0.6 | 0.8 | 0.8 | vLLM memory usage |

### Resource Allocation Guide

| GPU Count | vLLM GPUs | Training GPUs | Memory per GPU | Recommended Config |
|-----------|-----------|---------------|----------------|-------------------|
| 1 | 0.6 share | 0.4 share | 40GB+ | `single_gpu_config.yaml` |
| 2 | 1 (GPU 0) | 1 (GPU 1) | 24GB+ each | `custom_dataset_config.yaml` |
| 3 | 1 (GPU 0) | 2 (GPU 1-2) | 24GB+ each | Multi-GPU with DP |
| 4+ | 2 (GPU 0-1) | 2+ (GPU 2+) | 24GB+ each | Multi-GPU with TP+DP |

### Performance Optimization

1. **For Speed**: Increase `micro_batch_size`, reduce `gradient_accumulation_steps`
2. **For Memory**: Reduce `micro_batch_size`, increase `gradient_accumulation_steps`
3. **For Quality**: Increase `num_generations`, longer `sequence_len`
4. **For Efficiency**: Use dedicated GPUs, enable flash attention

## Quick Start Commands

### Dual GPU (Most Common)
```bash
# Terminal 1 - vLLM Server
./start_vllm_server.sh

# Terminal 2 - Training
./train_single_gpu.sh
```

### Single GPU Shared
```bash
# Terminal 1 - vLLM Server
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve single_gpu_config.yaml

# Terminal 2 - Training
./train_single_gpu_shared.sh
```

### Multi-GPU
```bash
# Terminal 1 - vLLM Server 1
CUDA_VISIBLE_DEVICES=0,1 axolotl vllm-serve multi_gpu_config.yaml

# Terminal 2 - Training
./train_multi_gpu.sh
```

---

**Note**: Always ensure vLLM servers are running and accessible before starting training. Monitor GPU memory usage with `nvidia-smi` to prevent out-of-memory errors.
