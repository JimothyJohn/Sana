#!/usr/bin/env python3
from cog import BasePredictor, Input, Path

import torch
from diffusers import SanaPipeline

from nunchaku.models.transformer_sana import NunchakuSanaTransformer2DModel

MODEL_CACHE = "/model-cache"


class Predictor(BasePredictor):
    def setup(self):
        transformer = NunchakuSanaTransformer2DModel.from_pretrained(f"{MODEL_CACHE}/mit-han-lab/svdq-int4-sana-1600m")
        self.pipe = SanaPipeline.from_pretrained(
            f"{MODEL_CACHE}/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            transformer=transformer,
            variant="bf16",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
    ) -> Path:
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
            generator=torch.Generator().manual_seed(42),
        ).images[0]
        image.save("sana.png")

        return Path("sana.png")
