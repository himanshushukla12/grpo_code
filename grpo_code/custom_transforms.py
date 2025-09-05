"""
Custom dataset transforms for instruction-response format datasets.
"""

# System prompt for code generation tasks
INSTRUCTION_RESPONSE_SYSTEM_PROMPT = """
You are a Python code generation assistant. Generate clean, well-structured Python code based on the given instructions. 

Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

The <answer> section should contain only the requested Python code without any additional explanations or markdown formatting.

You may use the following imports if needed:
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from typing import List, Dict, Tuple, Optional, Any
import sys
import os
import re
import json

If you choose to use any imports, include them at the beginning of your code inside the <answer> tags.

You may not use any external libraries or filesystem operations beyond the allowed imports.
"""


def instruction_response_transform(cfg, *args, **kwargs):
    """
    Transform function for instruction-response format datasets.
    
    This transform is designed for datasets where:
    - Each entry has an "instruction" field with the coding task/request
    - Each entry has a "response" field with the expected Python code
    
    The transform converts this to GRPO-compatible format without test cases,
    relying on code execution rewards for validation.
    
    Args:
        cfg: Axolotl configuration object
        
    Returns:
        tuple: (transform_function, dataset_config)
    """
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": example["instruction"] + "\n\n" + INSTRUCTION_RESPONSE_SYSTEM_PROMPT,
                }
            ],
            # For instruction-response datasets, we can create basic execution tests
            # or rely purely on code execution rewards
            "answers": [
                # Basic syntax check - try to compile the code
                f"compile({repr(example['response'])}, '<string>', 'exec')"
            ],
        }

    return transform_fn, {"remove_columns": ["instruction", "response"]}


def instruction_response_with_tests_transform(cfg, *args, **kwargs):
    """
    Enhanced transform function for instruction-response datasets that generates
    basic test cases automatically.
    
    This version creates more comprehensive test cases based on the instruction
    and response content.
    
    Args:
        cfg: Axolotl configuration object
        
    Returns:
        tuple: (transform_function, dataset_config)
    """
    def transform_fn(example, tokenizer=None):
        instruction = example["instruction"]
        response = example["response"]
        
        # Generate basic test cases
        test_cases = []
        
        # Always include compilation test
        test_cases.append(f"compile({repr(response)}, '<string>', 'exec')")
        
        # Try to extract function/class names and create basic tests
        import re
        
        # Look for function definitions
        func_matches = re.findall(r'def\s+(\w+)\s*\(', response)
        class_matches = re.findall(r'class\s+(\w+)\s*\(', response)
        
        # Add basic existence tests for functions and classes
        for func_name in func_matches:
            test_cases.append(f"exec({repr(response)}); assert callable({func_name})")
            
        for class_name in class_matches:
            test_cases.append(f"exec({repr(response)}); assert isinstance({class_name}, type)")
        
        # If no functions or classes found, just use compilation test
        if not func_matches and not class_matches:
            test_cases = [f"compile({repr(response)}, '<string>', 'exec')"]
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": instruction + "\n\n" + INSTRUCTION_RESPONSE_SYSTEM_PROMPT,
                }
            ],
            "answers": test_cases,
        }

    return transform_fn, {"remove_columns": ["instruction", "response"]}
