"""Text-to-Image pipeline — generate images from text prompts."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class Txt2ImgPipeline:
    """Text-to-Image generation pipeline.

    Wraps the HuggingFace StableDiffusionPipeline for text-to-image generation
    with a simplified interface.

    Example::

        pipe = Txt2ImgPipeline(model_id="runwayml/stable-diffusion-v1-5")
        images = pipe("a cat wearing a hat", num_inference_steps=30)
        images[0].save("cat.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
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
            from diffusers import StableDiffusionPipeline

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            logger.info("Txt2Img pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """Generate images from text.

        Args:
            prompt: Text prompt(s).
            negative_prompt: Negative prompt(s).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output width.
            height: Output height.
            seed: Random seed.
            num_images: Number of images per prompt.

        Returns:
            List of PIL Images.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images,
        )
        return result.images
