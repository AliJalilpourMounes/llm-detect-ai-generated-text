# src/llm_detector/utils.py
import numpy as np
import torch
import random
import os

def seed_everything(seed):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Can impact performance, set True if input sizes don't vary

def check_gpu():
    """Checks GPU availability and returns device info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("No GPU detected, running on CPU.")
    return device

def clean_memory():
    """Forces garbage collection and clears CUDA cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Forced garbage collection and cleared CUDA cache (if applicable).")