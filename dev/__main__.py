# AI-generated Sana image generation CLI
# Handles command line arguments and organizes outputs by date
import torch

# from diffusers import SanaPAGPipeline
import argparse
import json
import logging
from utils import *
from dev.pipeline import CustomSanaPipeline

# from app.sana_pipeline import SanaPAGPipeline

# Add logging configuration after imports
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def generate_image(pipe, prompt, seed, guidance_scale, pag_scale, steps, quality):
    """
    # AI-generated: Generate image with specified prompt and parameters
    # All parameters can be customized through CLI arguments
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latent, image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        num_inference_steps=steps,
        generator=generator,
    )[0]

    logging.info(f"Generated latent of shape: {latent.shape}")

    base_filename = f"sana_{datetime.now().strftime('%H%M%S')}"

    # Save image with shorter filename
    image_path = OUTPUT_DIR / f"{base_filename}.png"
    image[0].save(image_path)

    # Create metadata dictionary with all generation parameters
    metadata = {
        "quality": quality,
        "seed": seed,
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "pag_scale": pag_scale,
        "steps": steps,
        "image_filename": image_path.name,
    }

    # Save metadata to JSON file
    metadata_path = OUTPUT_DIR / f"{base_filename}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def setup_pipeline(quality="normal"):
    """
    # AI-generated: Initialize the Sana pipeline with optimal settings based on quality
    # Using appropriate model and precision settings for each quality level
    """
    config = MODEL_CONFIGS[quality]
    pipe = CustomSanaPipeline.from_pretrained(
        config["model_id"],
        variant=config["variant"],
        torch_dtype=config["dtype"],
        pag_applied_layers="transformer_blocks.8",
    )
    pipe.to(DEVICE)

    # Only apply bf16 conversion for normal/high quality models
    if config["variant"] == "bf16":
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
    return pipe


def main():
    # AI-generated: Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate images using Sana")
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Random seed(s) for generation. Can be single value, comma-separated list, or range (start:end:step)",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt(s) for image generation. Can be comma-separated list"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="normal",
        help="Model quality setting(s). Can be comma-separated list of: low, normal, high",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="5.0",
        help="Scale(s) for classifier-free guidance. Can be single value, comma-separated list, or range (start:end:step)",
    )
    parser.add_argument(
        "--pag-scale",
        type=str,
        default="2.0",
        help="Scale(s) for PAG. Can be single value, comma-separated list, or range (start:end:step)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="30",
        help="Number of inference steps. Can be single value, comma-separated list, or range (start:end:step)",
    )
    args = parser.parse_args()

    # Parse all arguments into lists
    seeds = parse_numeric_list(args.seed, int)
    prompts = parse_list_arg(args.prompt)
    cfgs = parse_numeric_list(args.cfg)
    pag_scales = parse_numeric_list(args.pag_scale)
    steps_list = parse_numeric_list(args.steps, int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(seeds, prompts, cfgs, pag_scales, steps_list))

    # Convert print to logging.info
    logging.info(f"Generating {len(param_combinations)} images...")

    # Check for high quality setting
    if args.quality == "high":
        logging.info(f"Skipping high quality combination (seed={seed}, prompt='{prompt}')")
        logging.info("🚧 4K Resolution Coming Soon! 🎨 ✨ 🔜 🎆")
        exit()

    # Generate images for all combinations
    pipe = setup_pipeline(args.quality)

    for seed, prompt, cfg, pag_scale, steps in param_combinations:
        generate_image(pipe, prompt, seed, cfg, pag_scale, steps, args.quality)


if __name__ == "__main__":
    main()
