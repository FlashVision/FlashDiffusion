"""Image-to-Image pipeline — transform images with text guidance."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class Img2ImgPipeline:
    """Image-to-Image generation pipeline.

    Takes an input image and transforms it according to a text prompt,
    controlling the amount of transformation via the strength parameter.

    Example::

        pipe = Img2ImgPipeline(model_id="runwayml/stable-diffusion-v1-5")
        images = pipe("watercolor painting", image="photo.jpg", strength=0.7)
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
            from diffusers import StableDiffusionImg2ImgPipeline

            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            logger.info("Img2Img pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[str, Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.7,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Transform an image with text guidance.

        Args:
            prompt: Text prompt describing the desired output.
            image: Input image (path or PIL Image).
            negative_prompt: Negative prompt.
            strength: How much to transform (0.0 = no change, 1.0 = full change).
            num_inference_steps: Denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed.

        Returns:
            List of PIL Images.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images
