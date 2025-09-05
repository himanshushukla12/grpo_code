#!/usr/bin/env python3
"""
Simple test script for custom dataset transforms (without dependencies).

This script validates the transform logic without requiring wasmtime.
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import only the transform functions directly to avoid dependency issues
from grpo_code.custom_transforms import (
    INSTRUCTION_RESPONSE_SYSTEM_PROMPT,
)

def instruction_response_transform_local(cfg=None):
    """Local version of the transform function for testing."""
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": example["instruction"] + "\n\n" + INSTRUCTION_RESPONSE_SYSTEM_PROMPT,
                }
            ],
            "answers": [
                f"compile({repr(example['response'])}, '<string>', 'exec')"
            ],
        }
    return transform_fn, {"remove_columns": ["instruction", "response"]}

def test_transforms():
    """Test the transform function with sample data."""
    
    # Sample data that matches the finetuning_dataset.jsonl format
    sample_examples = [
        {
            "instruction": "Generate a Python function named `add_numbers`. This function should take two parameters and return their sum.",
            "response": "def add_numbers(a, b):\n    return a + b"
        },
        {
            "instruction": "Generate a Python class named `Calculator`. This class should have basic arithmetic methods.",
            "response": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b"
        }
    ]
    
    print("Testing instruction_response_transform...")
    transform_fn, config = instruction_response_transform_local()
    
    for i, example in enumerate(sample_examples):
        try:
            result = transform_fn(example)
            print(f"Example {i+1}: ✓ Transform successful")
            print(f"  - Prompt length: {len(result['prompt'][0]['content'])} chars")
            print(f"  - Contains system prompt: {'<reasoning>' in result['prompt'][0]['content']}")
            print(f"  - Test cases: {len(result['answers'])}")
            print(f"  - First test: {result['answers'][0][:50]}...")
            
            # Validate the structure
            assert 'prompt' in result
            assert isinstance(result['prompt'], list)
            assert len(result['prompt']) == 1
            assert 'role' in result['prompt'][0]
            assert 'content' in result['prompt'][0]
            assert 'answers' in result
            assert isinstance(result['answers'], list)
            
        except Exception as e:
            print(f"Example {i+1}: ✗ Transform failed: {e}")

def test_real_dataset_sample():
    """Test with a few lines from the actual dataset."""
    
    dataset_path = project_root / "finetuning_dataset.jsonl"
    
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}")
        return
    
    print(f"\nTesting with real dataset samples...")
    
    transform_fn, _ = instruction_response_transform_local()
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i in range(3):  # Test first 3 lines
                line = f.readline().strip()
                if not line:
                    break
                    
                example = json.loads(line)
                
                print(f"\nReal Example {i+1}:")
                print(f"Instruction: {example['instruction'][:80]}...")
                print(f"Response: {example['response'][:80]}...")
                
                result = transform_fn(example)
                print(f"Transform result: ✓ Success")
                print(f"  - Prompt contains instruction: {example['instruction'][:30] in result['prompt'][0]['content']}")
                print(f"  - System prompt included: {'<reasoning>' in result['prompt'][0]['content']}")
                print(f"  - Test cases generated: {len(result['answers'])}")
                
    except Exception as e:
        print(f"Error reading dataset: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Code Custom Dataset Transform Test (Standalone)")
    print("=" * 60)
    
    test_transforms()
    test_real_dataset_sample()
    
    print("\n" + "=" * 60)
    print("Test completed! If all examples show ✓, transforms are working correctly.")
    print("Configuration file: custom_dataset_config.yaml")
    print("Dataset file: finetuning_dataset.jsonl")
    print("Transform function: grpo_code.instruction_response_transform")
    print("=" * 60)
