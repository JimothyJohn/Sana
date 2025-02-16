import torch
import numpy
import itertools
import os
from datetime import datetime
from pathlib import Path

DEVICE = "cuda"
HF_HUB_CACHE = f"{os.getenv('HOME')}/gigahawt/.cache/huggingface/hub"
# Create dated output directory
OUTPUT_DIR = Path("outputs") / datetime.now().strftime("%Y-%m-%d")

# AI-generated: Dictionary mapping quality settings to model IDs
MODEL_CONFIGS = {
    "low": {
        "model_id": "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "variant": None,  # Low quality model doesn't use bf16
        "dtype": torch.float16,
    },
    "normal": {
        "model_id": f"{HF_HUB_CACHE}/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots/e18f82ddb8233fa4d979c2613f41a3ca4c5fc730/",
        "variant": "bf16",
        "dtype": torch.bfloat16,
    },
    "high": {
        "model_id": "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
        "variant": "bf16",
        "dtype": torch.bfloat16,
    },
}


def parse_list_arg(value):
    """
    # AI-generated: Parse command line arguments that can be single values or lists
    # Converts comma-separated strings into lists of appropriate types
    """
    if "," not in str(value):
        return [value]
    return [x.strip() for x in value.split(",")]


def parse_numeric_list(value, convert_type=float):
    """
    # AI-generated: Parse numeric arguments that can be single values or lists
    # Handles both comma-separated values and range notation (start:end:step)
    """
    if ":" in str(value):
        start, end, *step = value.split(":")
        step = float(step[0]) if step else 1.0
        # Generate range of values
        values = list(numpy.arange(float(start), float(end) + step / 2, step))
        return [convert_type(v) for v in values]
    return [convert_type(v) for v in parse_list_arg(value)]
