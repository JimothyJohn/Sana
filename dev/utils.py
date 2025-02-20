import torch
import numpy
import itertools
import os
from datetime import datetime
from pathlib import Path
from dev.pipeline import CustomSanaPipeline
import json

DEVICE = "cuda"
HF_HUB_CACHE = f"{os.getenv('HF_HOME')}/hub"
OUTPUT_DIR = Path("outputs") / datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AI-generated: Dictionary mapping quality settings to model IDs
MODEL_CONFIGS = {
    "low": {
        "model_id": "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "variant": "fp16",  # Low quality model doesn't use bf16
        "dtype": torch.float16,
    },
    "normal": {
        # FP16
        # "model_id": "hf://Efficient-Large-Model/Sana_1600M_1024px",
        "dtype": torch.float16,
        "model_id": f"{HF_HUB_CACHE}/models--Efficient-Large-Model--Sana_1600M_1024px/snapshots/f69c052d762128f96927aa87fac6204978bddc8e/checkpoints/Sana_1600M_1024px.pth",
        "variant": "fp16",  # Low quality model doesn't use bf16
        # BF16
        # "model_id": f"{HF_HUB_CACHE}/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots/e18f82ddb8233fa4d979c2613f41a3ca4c5fc730/",
        # "dtype": torch.bfloat16,
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


def save_image(pipe, latent, filename):
    pipe.latents_to_image(latent).save(f"{OUTPUT_DIR}/{filename}.png")


def save_metadata(pipe, latent):
    # Create metadata dictionary with all generation parameters
    (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    metadata = {
        "quality": quality,
        "seed": seed,
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        # "pag_scale": pag_scale,
        "steps": steps,
        "image_filename": image_path.name,
    }

    # Save metadata to JSON file
    metadata_path = OUTPUT_DIR / f"metadata/{base_filename}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def generate_image(pipe, prompt, seed, guidance_scale, steps, quality):
    """
    # AI-generated: Generate image with specified prompt and parameters
    # All parameters can be customized through CLI arguments
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latent = pipe.generate_latents(
        prompt=prompt,
        guidance_scale=guidance_scale,
        # pag_guidance_scale=pag_scale,
        num_inference_steps=steps,
        generator=generator,
    )

    return latent


def setup_pipeline(quality="normal"):
    """
    # AI-generated: Initialize the Sana pipeline with optimal settings based on quality
    # Using appropriate model and precision settings for each quality level
    """
    config = MODEL_CONFIGS[quality]
    pipe = CustomSanaPipeline()
    pipe.from_pretrained(
        # f"Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        # model_path="hf://Efficient-Large-Model/Sana_1600M_1024px/Sana_1600M_1024px.pth",
        model_path=config["model_id"],
        # Only applicable in bf16
        # variant=config["variant"],
        # torch_dtype=config["dtype"],
        # pag_applied_layers="transformer_blocks.8",
    )
    pipe.to(DEVICE)

    # Only apply bf16 conversion for normal/high quality models
    if config["variant"] == "bf16":
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
    return pipe


def analyze_latents(latents: torch.Tensor, base_filename: str, output_dir: Path) -> dict:
    """
    AI-generated comment: Analyzes latent vectors and saves statistical information
    Args:
        latents: Input latent tensor to analyze
        base_filename: Base filename for the stats JSON
        output_dir: Directory to save the stats file
    Returns:
        Dictionary containing all computed statistics
    """
    # Move tensor to CPU and convert to float32 for calculations
    latents_cpu = latents.detach().cpu().float()

    # AI-generated comment: Compute basic statistics
    stats = {
        "shape": list(latents.shape),
        "min": float(latents_cpu.min()),
        "max": float(latents_cpu.max()),
        "mean": float(latents_cpu.mean()),
        "median": float(latents_cpu.median()),
        "std": float(latents_cpu.std()),
        # AI-generated comment: Compute distribution metrics
        "distribution": {
            "skewness": float(((latents_cpu - latents_cpu.mean()) ** 3).mean() / latents_cpu.std() ** 3),
            "kurtosis": float(((latents_cpu - latents_cpu.mean()) ** 4).mean() / latents_cpu.std() ** 4),
            "zero_count": int((latents_cpu == 0).sum()),
            "positive_count": int((latents_cpu > 0).sum()),
            "negative_count": int((latents_cpu < 0).sum()),
        },
        # AI-generated comment: Compute layer-wise statistics
        "channel_stats": {
            f"channel_{i}": {
                "mean": float(channel.mean()),
                "std": float(channel.std()),
                "min": float(channel.min()),
                "max": float(channel.max()),
            }
            for i, channel in enumerate(latents_cpu)
        },
        # AI-generated comment: Add timestamp
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save statistics to JSON file
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{base_filename}_stats.json"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
