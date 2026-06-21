"""Inpainting pipeline — fill masked regions with generated content."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InpaintingPipeline:
    """Inpainting pipeline for filling masked image regions.

    Example::

        pipe = InpaintingPipeline(model_id="runwayml/stable-diffusion-v1-5")
        images = pipe("a red car", image="photo.jpg", mask="mask.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from diffusers import StableDiffusionInpaintPipeline

            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            logger.info("Inpainting pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Fill masked regions of an image.

        Args:
            prompt: Text prompt describing what to generate in masked area.
            image: Input image (path or PIL Image).
            mask: Mask image (path or PIL Image) — white = inpaint, black = keep.
            negative_prompt: Negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: Guidance scale.
            seed: Random seed.

        Returns:
            List of PIL Images.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(mask, str):
            mask = Image.open(mask).convert("L")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images
