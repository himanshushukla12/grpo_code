
# GRPO Code: Group Relative Policy Optimization for Code Generation

A powerful reinforcement learning fine-tuning framework for large language models, specifically designed for code generation tasks. This project implements Group Relative Policy Optimization (GRPO) with interpreter feedback using WebAssembly sandboxing for safe code execution.

> [!NOTE] 
> Check out our [blog-post](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) for more detail and benchmarks!

## 🎯 Quick Start with Included Dataset

The project includes a comprehensive Python code generation dataset (`finetuning_dataset.jsonl`) ready for immediate use.

### Prerequisites
- CUDA-compatible GPU(s) with at least 16GB VRAM
- Python 3.8+
- 32GB+ RAM recommended for larger models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/axolotl-ai-cloud/grpo_code.git
cd grpo_code
```

2. Install the package and dependencies:
```bash
pip install -e .
pip install axolotl==0.8.0[vllm,flash-attn]
```

3. The Python WASM runtime is included (see [WASM Runtime Setup](#python-wasm-runtime) section for details)

### Using the Included Finetuning Dataset

1. **The dataset is already included** in the repository:
   - **File**: `finetuning_dataset.jsonl` 
   - **Size**: 132MB with 68,606 code generation examples
   - **Format**: Instruction-response pairs for Python functions and classes

2. **Start training immediately** with the pre-configured setup:

```bash
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=2,3 axolotl vllm-serve custom_dataset_config.yaml

# Terminal 2: Start training  
CUDA_VISIBLE_DEVICES=0,1 MAX_WORKERS=64 axolotl train custom_dataset_config.yaml --num-processes 2
```

3. **Monitor training progress**:
   - Training logs will show reward scores from the three reward functions
   - Models save at regular intervals defined in the config
   - Use Weights & Biases for detailed tracking (uncomment wandb settings in config)

### Training Configuration Overview

The `custom_dataset_config.yaml` is optimized for the included dataset:

```yaml
# Model and compute settings
base_model: Qwen/Qwen2.5-3B-Instruct
micro_batch_size: 32
gradient_accumulation_steps: 2
max_steps: 2500

# Dataset configuration  
datasets:
  - path: ./finetuning_dataset.jsonl
    type: grpo_code.instruction_response_transform

# Reward functions for code generation
trl:
  reward_funcs:
    - grpo_code.soft_format_reward_func      # Format validation (+0.25)
    - grpo_code.code_execution_reward_func   # Execution validation (+0.5/-0.25)
```

### Expected Training Results

With the included dataset, you can expect:
- **Improved code structure**: Better adherence to Python conventions
- **Syntax accuracy**: Reduced syntax errors in generated code  
- **Response formatting**: Consistent `<reasoning>` and `<answer>` structure
- **Training time**: ~2-4 hours on 4 A100 GPUs (2500 steps)

## 📖 Overview

This project enables reinforcement learning fine-tuning of language models using three specialized reward functions:

1. **Code Execution Reward** (`code_execution_reward_func`): Rewards models for generating syntactically correct and executable code
2. **Answer Execution Reward** (`answer_execution_reward_func`): Evaluates code correctness against test cases with accuracy-based scoring
3. **Soft Format Reward** (`soft_format_reward_func`): Ensures proper response formatting with `<reasoning>` and `<answer>` tags

### Key Features

- 🛡️ **Safe Code Execution**: Uses WebAssembly sandboxing to execute generated code safely
- ⚡ **Parallel Processing**: Multi-worker support for efficient reward function computation
- 🎯 **Multiple Reward Functions**: Comprehensive evaluation covering syntax, correctness, and formatting
- 🔧 **Configurable**: Extensive environment variable configuration for fine-tuning behavior
- 📊 **Dataset Agnostic**: Easy integration with custom datasets through transform functions

## 🏋️ Training

### Environment Variables

Configure the behavior of reward functions using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `WASM_FUEL` | Computation resources allocated to WASM environment | `1000000000` |
| `WASM_PATH` | Path to Python WASM runtime file | `./wasm/python-3.12.0.wasm` |
| `TASK_TIMEOUT` | Maximum execution time in seconds for code evaluation | `1` |
| `MAX_WORKERS` | Number of parallel workers for multiprocessing | `1` |
| `WORLD_SIZE` | Number of distributed training processes | `1` |

### Training Process

The training uses a two-stage approach with vLLM for efficient inference:

1. **Start vLLM Server** (for policy inference):
```bash
CUDA_VISIBLE_DEVICES=2,3 axolotl vllm-serve r1_acecode.yaml
```

2. **Launch Training** (in another terminal):
```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_WORKERS=64 axolotl train r1_acecode.yaml --num-processes 2
```

### Hardware Requirements

The example configuration uses 4 A100 GPUs:
- 2 GPUs for vLLM inference server
- 2 GPUs for training process

**Scaling Guidelines:**
- Adjust `CUDA_VISIBLE_DEVICES` for your available GPUs
- Modify `MAX_WORKERS` based on CPU cores (recommended: 16-64 workers)
- Update `micro_batch_size` and `gradient_accumulation_steps` in config for memory optimization

## 📊 Dataset Integration

### Using the Default AceCode Dataset

The project comes pre-configured with the AceCode-87K dataset:

```yaml
datasets:
  - path: axolotl-ai-co/AceCode-87K
    type: grpo_code.axolotl_acecode_transform
    split: train
```

### Using Custom Instruction-Response Datasets

This project now supports instruction-response format datasets like the included `finetuning_dataset.jsonl`. This format is ideal for general code generation tasks where you have coding instructions and expected implementations.

#### Dataset Format Requirements

Your JSONL dataset should contain entries with this structure:

```json
{
    "instruction": "Generate a Python function named `function_name`. This function should...",
    "response": "def function_name():\n    # implementation\n    pass"
}
```

**Example entry from finetuning_dataset.jsonl:**
```json
{
    "instruction": "Generate a Python function named `abort`. This function is located in the file `compile_py_script.py`. It has no identified dependencies. Please provide the complete, clean source code for the `abort` function.",
    "response": "def abort(msg):\n    print(msg)\n    sys.exit(1)"
}
```

#### Configuration for Custom Dataset

1. **Place your dataset file** in the project root (e.g., `finetuning_dataset.jsonl`)

2. **Create/modify your configuration file** (see `custom_dataset_config.yaml` for a complete example):

```yaml
datasets:
  - path: ./finetuning_dataset.jsonl
    type: grpo_code.instruction_response_transform
    # Alternative: use enhanced transform with automatic test generation
    # type: grpo_code.instruction_response_with_tests_transform
```

3. **Choose the appropriate transform:**

   - **`instruction_response_transform`**: Basic transform that creates simple compilation tests
   - **`instruction_response_with_tests_transform`**: Enhanced transform that automatically generates function/class existence tests

#### Transform Function Details

**Basic Transform (`instruction_response_transform`):**
- Creates compilation test: `compile(code, '<string>', 'exec')`
- Suitable for datasets focused on syntactic correctness
- Relies primarily on `code_execution_reward_func` and `soft_format_reward_func`

**Enhanced Transform (`instruction_response_with_tests_transform`):**
- Automatically detects function and class definitions
- Generates existence tests: `assert callable(function_name)`
- Creates type tests for classes: `assert isinstance(ClassName, type)`
- Better integration with `answer_execution_reward_func`

#### Complete Training Setup for Custom Dataset

1. **Use the provided configuration:**
```bash
# Start vLLM server
CUDA_VISIBLE_DEVICES=2,3 axolotl vllm-serve custom_dataset_config.yaml

# Start training (in another terminal)
CUDA_VISIBLE_DEVICES=0,1 MAX_WORKERS=64 axolotl train custom_dataset_config.yaml --num-processes 2
```

2. **Adjust reward functions** based on your dataset characteristics:

```yaml
trl:
  reward_funcs:
    - grpo_code.soft_format_reward_func        # Always recommended
    - grpo_code.code_execution_reward_func     # Basic syntax/execution validation
    # - grpo_code.answer_execution_reward_func # Enable if using enhanced transform
```

### Creating Custom Dataset Transforms

For datasets with different formats, create custom transform functions:

```python
def my_custom_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {
                    "role": "user", 
                    "content": example["your_question_field"] + "\n\n" + SYSTEM_PROMPT
                }
            ],
            "answers": example["your_test_cases_field"],  # List of assert statements
        }
    
    return transform_fn, {"remove_columns": ["your_question_field", "your_test_cases_field"]}
```

### Expected Response Format

The model is trained to respond in this specific format:

```xml
<reasoning>
I need to create a function that adds two numbers together.
This is a straightforward arithmetic operation.
</reasoning>
<answer>
def add_numbers(a, b):
    return a + b
</answer>
```

### Dataset Statistics

The included `finetuning_dataset.jsonl` contains:
- **68,606 instruction-response pairs**
- **132MB file size**
- **Focus**: Python function and class generation
- **Content**: Real-world code examples with detailed instructions

### Performance Optimization for Large Datasets

For large datasets like the 68K entry `finetuning_dataset.jsonl`:

```yaml
# Increase data loading workers
dataloader_num_workers: 4
dataloader_prefetch_factor: 64

# Enable data caching
dataset_prepared_path: /path/to/cache/prepared_data

# Optimize memory usage
micro_batch_size: 16  # Reduce if OOM
gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

## ⚙️ Configuration

### Main Configuration File (`r1_acecode.yaml`)

Key configuration sections:

#### Model Settings
```yaml
base_model: Qwen/Qwen2.5-3B-Instruct  # Base model to fine-tune
load_in_8bit: false                    # Memory optimization
load_in_4bit: false
torch_compile: true                    # Performance optimization
```

#### GRPO Settings
```yaml
rl: grpo                              # Use GRPO algorithm
trl:
  beta: 0.001                         # GRPO beta parameter
  use_vllm: true                      # Use vLLM for inference
  num_generations: 16                 # Samples per prompt
  max_completion_length: 512          # Max tokens in completion
  reward_funcs:                       # Reward function pipeline
    - grpo_code.soft_format_reward_func
    - grpo_code.code_execution_reward_func  
    - grpo_code.answer_execution_reward_func
```

#### Training Hyperparameters
```yaml
micro_batch_size: 32                  # Batch size per GPU
gradient_accumulation_steps: 2        # Gradient accumulation
learning_rate: 5.3e-6                # Learning rate
num_epochs: 1                        # Training epochs
max_steps: 2500                      # Maximum training steps
warmup_steps: 500                    # Warmup steps
```

#### Hardware Optimization
```yaml
bf16: true                           # Use bfloat16 precision
flash_attention: true                # Use Flash Attention
gradient_checkpointing: true         # Memory optimization
dataloader_num_workers: 2           # Data loading parallelism
```

## 🧪 Evaluation

The project includes evaluation scripts for HumanEval and MBPP benchmarks:

### Setup Evaluation Environment

```bash
cd eval_plus
pip install evalplus --upgrade
pip install -r requirements.txt
```

### Run Evaluation

```bash
bash test.sh /path/to/your/model 1 /path/to/output/dir
```

Parameters:
- `arg1`: Path to your trained model checkpoint
- `arg2`: Tensor parallel size (number of GPUs for inference)
- `arg3`: Output directory for results

### Evaluation Datasets

- **HumanEval**: 164 hand-written programming problems
- **HumanEval+**: Extended version with additional test cases
- **MBPP**: Mostly Basic Python Problems dataset
- **MBPP+**: Extended version of MBPP

## 🔧 Python WASM Runtime

This project uses Python 3.12.0 compiled to WebAssembly for safe code execution.

### Automatic Setup
The WASM files are included in the repository under `./wasm/`:
- `python-3.12.0.wasm` - Python runtime 
- `python-3.12.0.wasm.sha256sum` - Checksum for verification

### Verify Existing Installation

**Linux/macOS:**
```bash
sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
```

**Windows (PowerShell):**
```powershell
Get-FileHash -Algorithm SHA256 .\wasm\python-3.12.0.wasm
# Compare with hash in .sha256sum file
```

### Manual Download (if needed)

1. Download the Python WASM runtime:
```bash
curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm -o ./wasm/python-3.12.0.wasm
```

2. Download the SHA256 checksum:
```bash
curl -LO https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm.sha256sum -o ./wasm/python-3.12.0.wasm.sha256sum
```

3. Verify the download:
```bash
sha256sum -c ./wasm/python-3.12.0.wasm.sha256sum
```

## 🏗️ Architecture

### Core Components

1. **Reward Functions** (`grpo_code/rewards.py`)
   - `code_execution_reward_func`: Basic code execution validation
   - `answer_execution_reward_func`: Test case accuracy evaluation  
   - `soft_format_reward_func`: Response format validation

2. **Code Execution Engine** (`grpo_code/executor.py`, `grpo_code/wasm.py`)
   - WebAssembly-based Python runtime
   - Multi-process execution for parallel reward computation
   - Fuel-based resource limiting

3. **Dataset Transforms** (`grpo_code/transforms.py`)
   - Convert dataset formats to GRPO-compatible structure
   - Standardized prompt formatting with system instructions

4. **Parallel Processing** (`grpo_code/parallel_executor.py`)
   - Multi-worker code execution
   - Timeout handling and process pool management
   - Graceful error recovery

### Reward Function Details

#### Code Execution Reward
- **Positive Reward**: +0.5 for syntactically correct, executable code
- **Negative Reward**: -0.25 for syntax errors or execution failures
- **Purpose**: Encourages basic code correctness

#### Answer Execution Reward  
- **Scoring**: `(accuracy³) × 2` where accuracy = passed_tests / total_tests
- **Range**: 0.0 to 2.0
- **Purpose**: Heavily rewards high accuracy on test cases

#### Soft Format Reward
- **Reward**: +0.25 for correct `<reasoning>...</reasoning><answer>...</answer>` format
- **Purpose**: Ensures consistent response structure

## 🚀 Advanced Usage

### Custom Reward Functions

Create custom reward functions by following this pattern:

```python
def my_custom_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Custom reward function implementation.
    
    Args:
        completions: Model completions in chat format
        **kwargs: Additional arguments (answers, etc.)
    
    Returns:
        list[float]: Reward scores for each completion
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # Your custom scoring logic here
        reward = calculate_reward(content)
        rewards.append(reward)
    return rewards
```

Register in your config:
```yaml
trl:
  reward_funcs:
    - path.to.your.custom_reward_func
```

### Multi-GPU Training

For distributed training across multiple nodes:

```bash
# Node 1
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=8 RANK=0 \
axolotl train config.yaml --num-processes 4

# Node 2  
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=8 RANK=4 \
axolotl train config.yaml --num-processes 4
```

### Memory Optimization

For limited VRAM scenarios:

```yaml
# Enable quantization
load_in_8bit: true
# or
load_in_4bit: true

# Reduce batch sizes
micro_batch_size: 8
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: true
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `micro_batch_size`
   - Increase `gradient_accumulation_steps` 
   - Enable `gradient_checkpointing`
   - Use quantization (`load_in_8bit` or `load_in_4bit`)

2. **WASM Runtime Errors**
   - Verify WASM file integrity with checksum
   - Check `WASM_PATH` environment variable
   - Ensure sufficient `WASM_FUEL` allocation

3. **vLLM Connection Issues**
   - Verify vLLM server is running and accessible
   - Check `vllm_server_host` and `vllm_server_port` settings
   - Increase `vllm_server_timeout` if needed

4. **Slow Training**
   - Increase `MAX_WORKERS` for reward computation
   - Use faster storage (NVMe SSD)
   - Enable `torch_compile` for optimization

### Custom Dataset Troubleshooting

5. **Dataset Loading Errors**
   - Verify JSONL format: each line must be valid JSON
   - Check field names match transform function expectations
   - Ensure file path is correct (relative to config file)

6. **Transform Function Issues**
   - Verify the transform function is properly imported
   - Check that required fields (`instruction`, `response`) exist
   - Test transform on a small subset first

7. **Poor Training Performance with Custom Dataset**
   - Ensure data quality: remove malformed entries
   - Consider using `instruction_response_with_tests_transform` for better test coverage
   - Adjust reward function weights if needed
   - Verify examples are diverse and representative

8. **DTensor Mixed with Torch.Tensor Error**
   ```
   RuntimeError: aten.cat.default: got mixed torch.Tensor and DTensor
   ```
   **Solutions:**
   - Use `single_gpu_config.yaml` for single GPU setups
   - Set `torch_compile: false` in your configuration
   - Use separate GPUs for vLLM server and training process
   - Avoid `--num-processes 1` with distributed training configs

9. **Memory Issues with Large Datasets**
   ```yaml
   # Optimize for large datasets
   dataloader_num_workers: 4
   dataloader_prefetch_factor: 64
   dataset_prepared_path: /path/to/cache  # Enable caching
   micro_batch_size: 16  # Reduce batch size
   gradient_accumulation_steps: 8  # Increase to maintain effective batch size
   ```

10. **vLLM Server Connection Issues**
    - Ensure vLLM server is fully started before beginning training
    - Check that the server is accessible on the specified host/port
    - Verify tensor_parallel_size matches available GPUs for vLLM
    - Increase `vllm_server_timeout` if experiencing timeout errors

## 🎓 Complete Training Example

Here's a step-by-step guide to train a model using the included `finetuning_dataset.jsonl`:

### Step 1: Verify Setup

```bash
# Test that transforms work correctly
python simple_test_transforms.py

# Should show all examples with ✓ symbols
```

### Step 2: Choose Your Configuration

**For Single GPU Training:**
```bash
# Use the single GPU optimized config
cp single_gpu_config.yaml my_config.yaml
```

**For Multi-GPU Training:**
```bash
# Use the multi-GPU config (requires 2+ GPUs)
cp custom_dataset_config.yaml my_config.yaml
```

## Ubuntu Dual GPU Training Workflow

Your setup uses **two separate terminals running simultaneously**:

### Terminal 1: vLLM Inference Server (GPU 0)
```bash
# Start and keep running
chmod +x start_vllm_server.sh
./start_vllm_server.sh
```
**This terminal must stay open** - it provides inference for the training process.

### Terminal 2: GRPO Training (GPU 1) 
```bash
# Start after vLLM server is running
chmod +x train_single_gpu.sh
./train_single_gpu.sh
```
**This connects to the vLLM server** running in Terminal 1.

### Why Two Terminals?
- **Terminal 1**: Runs vLLM server continuously on GPU 0
- **Terminal 2**: Runs training on GPU 1, makes HTTP requests to Terminal 1's vLLM server
- **Communication**: Training sends prompts to vLLM via HTTP (localhost:8000)
- **Independence**: Each process uses its dedicated GPU without conflicts

### Step 3: Start vLLM Server (Terminal 1 - Keep Running)

**Ubuntu Dual GPU Setup (Recommended):**
```bash
# Terminal 1: Start vLLM server on GPU 0 and KEEP IT RUNNING
chmod +x start_vllm_server.sh
./start_vllm_server.sh

# Or manually:
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve custom_dataset_config.yaml
```

⚠️ **Important**: Do NOT close this terminal! The vLLM server must keep running.

**Single GPU Setup:**
```bash
# Terminal 1: Start the inference server (shared GPU)
CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve single_gpu_config.yaml
```

**Multi-GPU Setup:**
```bash
# Terminal 1: Start the inference server on separate GPUs
CUDA_VISIBLE_DEVICES=2,3 axolotl vllm-serve custom_dataset_config.yaml
```

Wait for the server to start (you'll see "Uvicorn running on http://0.0.0.0:8000").

### Step 4: Start Training (Terminal 2 - While vLLM Runs)

**Ubuntu Dual GPU Training (GPU 0: vLLM, GPU 1: Training):**
```bash
# Terminal 2: Start training on GPU 1 (connects to Terminal 1's vLLM)
chmod +x train_single_gpu.sh
./train_single_gpu.sh

# Or manually:
CUDA_VISIBLE_DEVICES=1 MAX_WORKERS=64 accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 1 --num_machines 1 --machine_rank 0 \
  -m axolotl.cli.train custom_dataset_config.yaml

# Or use Python launcher with health checks:
python launch_training.py
```

💡 **Note**: Training process sends HTTP requests to vLLM server (Terminal 1) for inference.

**Single GPU Training:**
```bash
# Terminal 2: Train on a separate GPU or use CPU-only training
CUDA_VISIBLE_DEVICES=1 MAX_WORKERS=32 axolotl train single_gpu_config.yaml

# Alternative: If you have only one GPU, stop vLLM and use CPU for rewards
# CUDA_VISIBLE_DEVICES=0 MAX_WORKERS=16 axolotl train single_gpu_config.yaml
```

**Multi-GPU Training:**
```bash
# Terminal 2: Start the RL training on separate GPUs
CUDA_VISIBLE_DEVICES=0,1 MAX_WORKERS=64 axolotl train custom_dataset_config.yaml --num-processes 2
```

### Step 5: Monitor Training

Expected output during training:
```
[INFO] Loading dataset: ./finetuning_dataset.jsonl (68,606 examples)
[INFO] Using transform: grpo_code.instruction_response_transform
[INFO] Starting GRPO training...
[INFO] Step 1/1000 | Reward: format=0.85, execution=0.45, total=1.30
[INFO] Step 10/1000 | Reward: format=0.90, execution=0.52, total=1.42
```

### Troubleshooting the DTensor Error

If you see the error `RuntimeError: aten.cat.default: got mixed torch.Tensor and DTensor`, or `RuntimeError: Invalid device argument : did you call init?`, try:

1. **Use the single GPU launcher script**:
```bash
# Make the script executable
chmod +x train_single_gpu.sh
./train_single_gpu.sh

# Or use the Python launcher
python launch_training.py
```

2. **Use explicit accelerate configuration**:
```bash
# Use the provided accelerate config
CUDA_VISIBLE_DEVICES=1 MAX_WORKERS=32 accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 1 \
  -m axolotl.cli.train custom_dataset_config.yaml
```

3. **Force single GPU mode**:
```bash
# Bypass distributed training entirely
CUDA_VISIBLE_DEVICES=1 MAX_WORKERS=32 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  python -m axolotl.cli.train custom_dataset_config.yaml
```

4. **Alternative: Use CPU for reward functions**:
```bash
# If you have memory constraints, use CPU for reward computation
CUDA_VISIBLE_DEVICES=1 MAX_WORKERS=16 FORCE_CPU_REWARDS=1 \
  python -m axolotl.cli.train custom_dataset_config.yaml
```

### Fix for "Invalid device argument" Error

This error occurs when CUDA context isn't properly initialized. Solutions:

1. **Clear accelerate cache**:
```bash
rm -rf ~/.cache/huggingface/accelerate/
```

2. **Use explicit device initialization**:
```python
# Run this first to test CUDA
python -c "import torch; torch.cuda.init(); print(f'Device: {torch.cuda.current_device()}')"
```

3. **Check GPU availability**:
```bash
nvidia-smi
# Ensure the GPU you're targeting (CUDA_VISIBLE_DEVICES=1) exists
```

### Step 6: Evaluate Results

After training completes, test your model:

```bash
# Use the evaluation scripts in eval_plus/
cd eval_plus
bash test.sh /path/to/your/trained/model 1 ./results
```

### Expected Training Timeline

**Single GPU:**
- **Initialization**: 5-10 minutes
- **Training**: 1-2 hours for 1000 steps
- **Memory usage**: ~16-20GB GPU memory

**Multi-GPU:**
- **Initialization**: 5-10 minutes  
- **Training**: 2-4 hours for 2500 steps
- **Memory usage**: ~60GB total GPU memory

### Key Metrics to Watch

1. **Format Reward**: Should quickly reach ~0.9+ (model learns to use proper `<reasoning>`/`<answer>` tags)
2. **Execution Reward**: Gradual improvement from ~0.3 to 0.6+ (fewer syntax errors)
3. **Combined Score**: Target >1.5 for good performance

### Configuration Comparison

| Feature | Single GPU | Multi-GPU |
|---------|------------|-----------|
| GPUs Required | 1-2 | 3-4 |
| Batch Size | 8 | 32 |
| Gradient Accumulation | 8 | 2 |
| Training Steps | 1000 | 2500 |
| vLLM Generations | 8 | 16 |
| Training Time | 1-2 hours | 2-4 hours |

## 📈 Expected Results

After training with the included dataset, your model should demonstrate:

### Improved Code Generation
- **Syntax accuracy**: 90%+ syntactically correct Python code
- **Structure quality**: Better function/class organization
- **Documentation**: More comprehensive docstrings and comments

### Consistent Formatting
- **Response structure**: Proper `<reasoning>` and `<answer>` tags
- **Code cleanliness**: Better indentation and style
- **Error reduction**: Fewer runtime and compilation errors

### Training Metrics
- **Final format reward**: 0.90+ (excellent adherence to response format)
- **Final execution reward**: 0.60+ (most code executes without errors)
- **Training stability**: Steady improvement without significant divergence

## 📋 Requirements

### System Requirements
- **GPU**: CUDA-compatible with 16GB+ VRAM (A100/H100 recommended)
- **CPU**: 16+ cores recommended for parallel reward computation  
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for models and datasets

### Software Dependencies
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
- See `pyproject.toml` for complete dependency list

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{grpo_code,
  title={GRPO Code: Group Relative Policy Optimization for Code Generation},
  author={Axolotl AI},
  year={2024},
  url={https://github.com/axolotl-ai-cloud/grpo_code}
}
```

## 🔗 Related Resources

- [Axolotl Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [EvalPlus Benchmark](https://github.com/evalplus/evalplus)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [WebAssembly Language Runtimes](https://github.com/vmware-labs/webassembly-language-runtimes)
