"""SDXL text-to-image pipeline with base + refiner support."""

import logging
from typing import List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SDXLPipeline:
    """SDXL base model pipeline for high-resolution image generation.

    Supports 1024x1024 generation with dual text encoders and
    SDXL-specific micro-conditioning.

    Args:
        model_id: SDXL base model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
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
            from diffusers import StableDiffusionXLPipeline

            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            ).to(self.device)
            logger.info("SDXL pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
        denoising_end: Optional[float] = None,
        output_type: str = "pil",
    ) -> List[Image.Image]:
        """Generate images with SDXL.

        Args:
            prompt: Primary text prompt.
            prompt_2: Secondary prompt for second text encoder.
            negative_prompt: Primary negative prompt.
            negative_prompt_2: Secondary negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            width: Output width (recommended: 1024).
            height: Output height (recommended: 1024).
            seed: Random seed.
            num_images: Images per prompt.
            denoising_end: End point for base model (for refiner handoff).
            output_type: Output format ("pil" or "latent").

        Returns:
            List of PIL Images (or latents if output_type="latent").
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images,
            output_type=output_type,
        )
        if denoising_end is not None:
            kwargs["denoising_end"] = denoising_end

        result = self.pipe(**kwargs)
        return result.images


class SDXLRefinerPipeline:
    """SDXL refiner pipeline for enhancing base model outputs.

    Takes latent output from the base model and applies additional
    denoising for improved details and coherence.

    Args:
        model_id: SDXL refiner model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
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
            from diffusers import StableDiffusionXLImg2ImgPipeline

            self._pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
            ).to(self.device)
            logger.info("SDXL refiner loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.Tensor, List[Image.Image]],
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.3,
        denoising_start: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Refine images from the base model.

        Args:
            prompt: Text prompt.
            image: Base model output (latents or PIL images).
            num_inference_steps: Refiner denoising steps.
            guidance_scale: CFG scale for refiner.
            strength: Refinement strength.
            denoising_start: Start point (matching base model's denoising_end).
            seed: Random seed.

        Returns:
            List of refined PIL Images.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        if denoising_start is not None:
            kwargs["denoising_start"] = denoising_start

        result = self.pipe(**kwargs)
        return result.images
