"""ImageGenerator — high-level text-to-image generation with sensible defaults."""

import logging
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ImageGenerator:
    """High-level image generator with production-ready defaults.

    Wraps FlashDiffusion with sensible defaults for prompt engineering,
    negative prompts, and image quality settings.

    Example::

        from flashdiffusion.solutions import ImageGenerator

        gen = ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")
        image = gen("a beautiful sunset over mountains", seed=42)
        image.save("sunset.png")
    """

    DEFAULT_NEGATIVE = (
        "blurry, low quality, low resolution, ugly, deformed, bad anatomy, watermark, text, signature, cropped"
    )

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        default_steps: int = 30,
        default_guidance: float = 7.5,
        default_width: int = 512,
        default_height: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self.default_width = default_width
        self.default_height = default_height
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from flashdiffusion.engine.predictor import Predictor

            self._predictor = Predictor(model_id=self.model_id, device=self.device)
        return self._predictor

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a single image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: What to avoid (uses defaults if None).
            steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output width.
            height: Output height.
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        images = self.predictor.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or self.DEFAULT_NEGATIVE,
            num_inference_steps=steps or self.default_steps,
            guidance_scale=guidance_scale or self.default_guidance,
            width=width or self.default_width,
            height=height or self.default_height,
            seed=seed,
        )
        return images[0]

    def batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images for multiple prompts.

        Args:
            prompts: List of text prompts.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated PIL Images.
        """
        return [self(prompt, **kwargs) for prompt in prompts]
