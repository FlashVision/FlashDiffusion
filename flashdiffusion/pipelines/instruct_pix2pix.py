"""InstructPix2Pix — instruction-guided image editing pipeline.

Edits images based on text instructions without requiring explicit
source and target descriptions.

Reference: https://arxiv.org/abs/2211.09800
"""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InstructPix2PixPipeline:
    """Instruction-guided image editing pipeline.

    Takes an input image and a text instruction describing the desired edit,
    then produces the edited image. Uses a modified Stable Diffusion model
    trained on instruction-image pairs.

    Example::

        pipe = InstructPix2PixPipeline()
        result = pipe(
            prompt="make it a watercolor painting",
            image="photo.jpg",
        )
        result[0].save("edited.png")

    Args:
        model_id: InstructPix2Pix model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
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
            from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

            self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.device)
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._pipe.scheduler.config,
            )
            logger.info("InstructPix2Pix loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[str, Image.Image],
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """Edit an image based on text instruction.

        Args:
            prompt: Text instruction describing the edit.
            image: Input image to edit (path or PIL Image).
            num_inference_steps: Denoising steps.
            guidance_scale: Text guidance scale (higher = follow instruction more).
            image_guidance_scale: Image guidance scale (higher = stay closer to input).
            seed: Random seed.
            num_images: Number of output images.

        Returns:
            List of edited PIL Images.
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
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
        )
        return result.images

    def batch_edit(
        self,
        prompts: List[str],
        images: List[Union[str, Image.Image]],
        **kwargs,
    ) -> List[List[Image.Image]]:
        """Edit multiple images with corresponding instructions.

        Args:
            prompts: List of edit instructions.
            images: List of input images.

        Returns:
            List of lists of edited images.
        """
        results = []
        for prompt, image in zip(prompts, images):
            result = self(prompt=prompt, image=image, **kwargs)
            results.append(result)
        return results
