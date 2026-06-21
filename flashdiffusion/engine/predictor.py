"""FlashDiffusion Predictor — high-level image generation interface."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

from flashdiffusion.models.flash_diffusion import FlashDiffusion

logger = logging.getLogger(__name__)


class Predictor:
    """High-level generation wrapper for FlashDiffusion.

    Example::

        from flashdiffusion import Predictor

        pred = Predictor(model_id="runwayml/stable-diffusion-v1-5")
        images = pred.generate("a castle on a hill, sunset")
        images[0].save("castle.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        scheduler: Optional[str] = None,
        lora_weights: Optional[str] = None,
    ):
        self.model = FlashDiffusion(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            scheduler=scheduler,
        )

        if lora_weights is not None:
            self._load_lora(lora_weights)

    def _load_lora(self, lora_path: str):
        """Load LoRA weights into the UNet."""
        pipe = self.model.pipe
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(lora_path)
            logger.info("LoRA weights loaded: %s", lora_path)
        else:
            logger.warning("Pipeline does not support load_lora_weights")

    def generate(
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
        """Generate images from text prompt.

        Args:
            prompt: Text prompt(s).
            negative_prompt: Negative prompt(s).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output width.
            height: Output height.
            seed: Random seed for reproducibility.
            num_images: Number of images to generate.

        Returns:
            List of PIL Images.
        """
        return self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            num_images=num_images,
        )

    def generate_grid(
        self,
        prompts: List[str],
        cols: int = 4,
        **kwargs,
    ) -> Image.Image:
        """Generate a grid of images from multiple prompts.

        Args:
            prompts: List of text prompts.
            cols: Number of columns in the grid.
            **kwargs: Additional generation arguments.

        Returns:
            PIL Image grid.
        """
        from flashdiffusion.utils.image_utils import make_image_grid

        images = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            images.extend(result)

        return make_image_grid(images, cols=cols)
