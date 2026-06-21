"""ControlNet pipeline — conditioned generation with structural control."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

from flashdiffusion.models.controlnet import ControlNetWrapper

logger = logging.getLogger(__name__)


class ControlNetPipeline:
    """ControlNet-conditioned image generation pipeline.

    Example::

        pipe = ControlNetPipeline(controlnet_type="canny")
        images = pipe("a beautiful house", control_image="edges.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_type: str = "canny",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.controlnet = ControlNetWrapper(controlnet_type=controlnet_type, torch_dtype=self.torch_dtype)
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from diffusers import StableDiffusionControlNetPipeline
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=self.controlnet.model,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            logger.info("ControlNet pipeline loaded: %s + %s", self.model_id, self.controlnet.controlnet_type)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        control_image: Union[str, Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images conditioned on a control signal."""
        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )
        return result.images
