"""StyleTransfer — apply artistic styles via img2img pipeline."""

import logging
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class StyleTransfer:
    """Apply artistic styles to images using the img2img pipeline.

    Example::

        from flashdiffusion.solutions import StyleTransfer

        transfer = StyleTransfer(model_id="runwayml/stable-diffusion-v1-5")
        styled = transfer(
            image="photo.jpg",
            style_prompt="van gogh oil painting style",
            strength=0.6,
        )
        styled.save("styled.png")
    """

    STYLE_PRESETS = {
        "oil_painting": "oil painting, thick brush strokes, rich colors, canvas texture",
        "watercolor": "watercolor painting, soft edges, transparent washes, paper texture",
        "pencil_sketch": "detailed pencil sketch, graphite drawing, fine lines, shading",
        "anime": "anime style, cel shading, vibrant colors, clean lines",
        "pixel_art": "pixel art, 16-bit style, retro game aesthetic",
        "photorealistic": "photorealistic, highly detailed, 8k, professional photography",
    }

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from flashdiffusion.pipelines import Img2ImgPipeline
            self._pipe = Img2ImgPipeline(model_id=self.model_id, device=self.device)
        return self._pipe

    def __call__(
        self,
        image: Union[str, Image.Image],
        style_prompt: str = "oil painting style",
        strength: float = 0.6,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Apply a style to an input image.

        Args:
            image: Input image path or PIL Image.
            style_prompt: Text describing the target style (or a preset name).
            strength: Transformation strength (0.0-1.0).
            num_inference_steps: Denoising steps.
            guidance_scale: Guidance scale.
            seed: Random seed.

        Returns:
            Styled PIL Image.
        """
        prompt = self.STYLE_PRESETS.get(style_prompt, style_prompt)

        images = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return images[0]
