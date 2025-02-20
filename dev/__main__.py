# AI-generated Sana image generation CLI
# Handles command line arguments and organizes outputs by date
import torch
import argparse
import json
import logging
from utils import *
from pipeline import *

# Add logging configuration after imports
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


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
    """
    parser.add_argument(
        "--pag-scale",
        type=str,
        default="2.0",
        help="Scale(s) for PAG. Can be single value, comma-separated list, or range (start:end:step)",
    )
    """
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
    # pag_scales = parse_numeric_list(args.pag_scale)
    steps_list = parse_numeric_list(args.steps, int)

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(seeds, prompts, cfgs, steps_list))

    # Convert print to logging.info
    logging.info(f"Generating {len(param_combinations)} images...")

    # Check for high quality setting
    if args.quality == "high":
        logging.info(f"Skipping high quality combination (seed={seed}, prompt='{prompt}')")
        logging.info("🚧 4K Resolution Coming Soon! 🎨 ✨ 🔜 🎆")
        exit()

    # Generate images for all combinations
    pipe = setup_pipeline(args.quality)

    for seed, prompt, cfg, steps in param_combinations:
        latent = generate_image(pipe, prompt, seed, cfg, steps, args.quality)
        latent_two = generate_image(pipe, "greenhorn", seed, cfg, steps, args.quality)
        create_transition_video(
            pipe=pipe,
            latents_one=latent,
            latents_two=latent_two,
            output_filename=f"transition_{datetime.now().strftime('%H%M%S')}",
            fps=60,
            steps=60,  # Adjust for smoother/faster transitions
            output_dir=OUTPUT_DIR,
        )
        stats = analyze_latents(latent, f"sana_{datetime.now().strftime('%H%M%S')}", OUTPUT_DIR)
        save_image(pipe, latent, f"sana_{datetime.now().strftime('%H%M%S')}")
        save_image(pipe, multiply_latents(latent, 2), f"sana_multiply_{datetime.now().strftime('%H%M%S')}")
        save_image(pipe, offset_latents(latent, 2), f"sana_offset_{datetime.now().strftime('%H%M%S')}")


if __name__ == "__main__":
    main()
