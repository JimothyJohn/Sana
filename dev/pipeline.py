import warnings

warnings.filterwarnings("ignore")  # ignore warning
import torch
import pyrallis
import yaml
from PIL import Image

from app.sana_pipeline import SanaPipeline

from diffusion import DPMS, FlowEuler
from diffusion.model.builder import vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from app.sana_pipeline import guidance_type_select, classify_height_width_bin
from typing import Optional, Tuple
import subprocess
import shutil
from pathlib import Path


def latents_to_image(latents, config, vae, ori_width, ori_height):
    with torch.no_grad():
        sample = vae_decode(config.vae.vae_type, vae, latents)

    sample = resize_and_crop_tensor(sample, ori_width, ori_height)
    # AI-generated comment: Convert tensor to PIL Image for saving
    # First clamp values between 0 and 1, then convert to uint8 range (0-255)
    sample = (sample.clamp(-1, 1) + 1) / 2
    sample = (sample * 255).round().to(torch.uint8)
    # AI-generated comment: Handle 4D tensor (B,C,H,W) by selecting first image
    sample = sample[0] if sample.dim() == 4 else sample
    # Reshape from (C,H,W) to (H,W,C) and convert to PIL
    sample = sample.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(sample)


class CustomSanaPipeline(SanaPipeline):
    def latents_to_image(self, latents):
        with torch.no_grad():
            sample = vae_decode(self.config.vae.vae_type, self.vae, latents)

        sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
        # AI-generated comment: Convert tensor to PIL Image for saving
        # First clamp values between 0 and 1, then convert to uint8 range (0-255)
        sample = (sample.clamp(-1, 1) + 1) / 2
        sample = (sample * 255).round().to(torch.uint8)
        # AI-generated comment: Handle 4D tensor (B,C,H,W) by selecting first image
        sample = sample[0] if sample.dim() == 4 else sample
        # Reshape from (C,H,W) to (H,W,C) and convert to PIL
        sample = sample.permute(1, 2, 0).cpu().numpy()

        return Image.fromarray(sample)

    def generate_latents(
        self,
        prompt=None,
        height=1024,
        width=1024,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=5,
        pag_guidance_scale=2.5,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
    ):
        self.ori_height, self.ori_width = height, width
        self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        # 1. pre-compute negative embedding
        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        for prompt in prompts:
            # data prepare
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
            )
            for _ in range(num_images_per_prompt):
                with torch.no_grad():
                    prompts.append(
                        prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip()
                    )

                    # prepare text feature
                    if not self.config.text_encoder.chi_prompt:
                        max_length_all = self.config.text_encoder.model_max_length
                        prompts_all = prompts
                    else:
                        chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                        prompts_all = [chi_prompt + prompt for prompt in prompts]
                        num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_chi_prompt_tokens + self.config.text_encoder.model_max_length - 2
                        )  # magic number 2: [bos], [_]

                    caption_token = self.tokenizer(
                        prompts_all,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device=self.device)
                    select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))
                    caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                        :, :, select_index
                    ].to(self.weight_dtype)
                    emb_masks = caption_token.attention_mask[:, select_index]
                    null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

                    n = len(prompts)
                    if latents is None:
                        z = torch.randn(
                            n,
                            self.config.vae.vae_latent_dim,
                            self.latent_size_h,
                            self.latent_size_w,
                            generator=generator,
                            device=self.device,
                            dtype=self.weight_dtype,
                        )
                    else:
                        z = latents.to(self.weight_dtype).to(self.device)
                    model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
                    if self.vis_sampler == "flow_euler":
                        flow_solver = FlowEuler(
                            self.model,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=guidance_scale,
                            model_kwargs=model_kwargs,
                        )
                        sample = flow_solver.sample(
                            z,
                            steps=num_inference_steps,
                        )
                    elif self.vis_sampler == "flow_dpm-solver":
                        scheduler = DPMS(
                            self.model,
                            condition=caption_embs,
                            uncondition=null_y,
                            guidance_type=self.guidance_type,
                            cfg_scale=guidance_scale,
                            pag_scale=pag_guidance_scale,
                            pag_applied_layers=self.config.model.pag_applied_layers,
                            model_type="flow",
                            model_kwargs=model_kwargs,
                            schedule="FLOW",
                        )
                        scheduler.register_progress_bar(self.progress_fn)
                        sample = scheduler.sample(
                            z,
                            steps=num_inference_steps,
                            order=2,
                            skip_type="time_uniform_flow",
                            method="multistep",
                            flow_shift=self.flow_shift,
                        )

            latent = sample.to(self.weight_dtype)
            # image = latents_to_image(latent, self.config, self.vae, self.ori_width, self.ori_height)
            # samples.append(image)

            return latent

        return samples


# AI-generated comment: Helper functions for latent vector manipulation


def multiply_latents(latents: torch.Tensor, multiple: float) -> torch.Tensor:
    """
    AI-generated comment: Multiplies all values in the latent tensor by a scalar
    Args:
        latents: Input latent tensor
        multiple: Scalar multiplication factor
    Returns:
        Scaled latent tensor
    """
    return latents * multiple


def transition_latents(latents_one: torch.Tensor, latents_two: torch.Tensor, steps: int) -> torch.Tensor:
    """
    AI-generated comment: Creates a smooth transition between two latent vectors
    Args:
        latents_one: Starting latent tensor
        latents_two: Ending latent tensor
        steps: Number of interpolation steps
    Returns:
        Tensor containing all transition steps [steps, *latent_dims]
    """
    # Create interpolation weights
    weights = torch.linspace(0, 1, steps, device=latents_one.device)
    # Add dimensions to match latent tensor shape
    weights = weights.view(-1, *([1] * len(latents_one.shape)))

    # Interpolate between tensors
    transitions = latents_one * (1 - weights) + latents_two * weights
    return transitions


def create_transition_video(
    pipe,
    latents_one: torch.Tensor,
    latents_two: torch.Tensor,
    output_filename: str,
    fps: int = 60,
    steps: int = 60,
    output_dir: Path = "outputs/",
) -> Path:
    """
    AI-generated comment: Creates a smooth video transition between two latent states
    Args:
        pipe: The Sana pipeline for image generation
        latents_one: Starting latent tensor
        latents_two: Ending latent tensor
        output_filename: Base name for the output video (without extension)
        fps: Frames per second for the output video
        steps: Number of interpolation steps
        output_dir: Directory to save the video and frames
    Returns:
        Path to the generated video file
    """

    # AI-generated comment: Create temporary directory for frames
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate transition frames
        transitions = transition_latents(latents_one, latents_two, steps)

        # AI-generated comment: Save each frame
        for i, latent in enumerate(transitions):
            # Ensure latent is in the correct dtype for the model
            latent = latent.to(dtype=pipe.weight_dtype)
            frame = pipe.latents_to_image(latent)
            frame_path = temp_dir / f"frame_{i:04d}.png"
            frame.save(frame_path)

        # AI-generated comment: Use ffmpeg to create video
        video_path = output_dir / f"{output_filename}.mp4"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",  # High quality encoding
            "-crf",
            "18",  # High quality (0-51, lower is better)
            "-pix_fmt",
            "yuv420p",  # Widely compatible pixel format
            str(video_path),
        ]

        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        return video_path

    finally:
        # AI-generated comment: Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def offset_latents(latents: torch.Tensor, offset: float) -> torch.Tensor:
    """
    AI-generated comment: Adds a constant offset to all values in the latent tensor
    Args:
        latents: Input latent tensor
        offset: Value to add to all elements
    Returns:
        Offset latent tensor
    """
    return latents + offset


def mix_latents(latents_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """
    AI-generated comment: Combines multiple latent vectors with weighted averaging
    Args:
        latents_list: List of latent tensors to combine
        weights: List of weights for each latent tensor (should sum to 1)
    Returns:
        Combined latent tensor
    """
    if len(latents_list) != len(weights):
        raise ValueError("Number of latents must match number of weights")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1")

    result = sum(l * w for l, w in zip(latents_list, weights))
    return result


def add_noise_to_latents(
    latents: torch.Tensor, noise_strength: float = 0.1, seed: Optional[int] = None
) -> torch.Tensor:
    """
    AI-generated comment: Adds controlled random noise to latent vectors
    Args:
        latents: Input latent tensor
        noise_strength: Standard deviation of noise (default: 0.1)
        seed: Random seed for reproducibility (optional)
    Returns:
        Latent tensor with added noise
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(latents) * noise_strength
    return latents + noise


def rotate_latents_2d(latents: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """
    AI-generated comment: Rotates latent vectors in 2D latent space
    Note: This assumes the latent space has meaningful geometric properties
    Args:
        latents: Input latent tensor
        angle_degrees: Rotation angle in degrees
    Returns:
        Rotated latent tensor
    """
    angle_rad = torch.tensor(angle_degrees * torch.pi / 180.0)
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    # Create rotation matrix
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=latents.device)

    # Reshape latents to 2D, rotate, and reshape back
    original_shape = latents.shape
    latents_2d = latents.reshape(-1, 2)
    rotated_2d = torch.matmul(latents_2d, rotation_matrix)

    return rotated_2d.reshape(original_shape)
