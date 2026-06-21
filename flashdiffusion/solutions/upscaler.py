"""Upscaler — image upscaling via diffusion models."""

import logging
from typing import Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class Upscaler:
    """Upscale images using diffusion-based super-resolution.

    Example::

        from flashdiffusion.solutions import Upscaler

        upscaler = Upscaler()
        upscaled = upscaler(image="low_res.jpg", prompt="high quality photo")
        upscaled.save("high_res.png")
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def _load_pipeline(self):
        from diffusers import StableDiffusionUpscalePipeline

        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        logger.info("Upscaler pipeline loaded: %s", self.model_id)
        return pipe

    def __call__(
        self,
        image: Union[str, Image.Image],
        prompt: str = "high quality, detailed, sharp",
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Upscale an image.

        Args:
            image: Input low-resolution image (path or PIL Image).
            prompt: Text prompt to guide upscaling quality.
            num_inference_steps: Denoising steps.
            guidance_scale: Guidance scale.
            seed: Random seed.

        Returns:
            Upscaled PIL Image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
