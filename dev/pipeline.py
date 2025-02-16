import warnings

warnings.filterwarnings("ignore")  # ignore warning
import torch

from diffusers import SanaPAGPipeline

from diffusion import DPMS, FlowEuler
from diffusion.model.builder import vae_decode
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor


class CustomSanaPipeline(SanaPAGPipeline):
    @torch.inference_mode()
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
                prompts.append(prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip())

            with torch.no_grad():
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
                    )
                else:
                    z = latents.to(self.device)
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

            latent_sample = sample.to(self.vae_dtype)
            with torch.no_grad():
                latent_sample = vae_decode(self.config.vae.vae_type, self.vae, sample)

            sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

            return latent_sample, sample

        return latent_sample, samples
