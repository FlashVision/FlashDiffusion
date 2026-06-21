"""FlashDiffusion — main pipeline wrapper around HuggingFace diffusers.

Provides a unified interface for loading and running Stable Diffusion,
SDXL, and FLUX models with configurable schedulers and generation parameters.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FlashDiffusion:
    """High-level wrapper for diffusion model pipelines.

    Example::

        from flashdiffusion import FlashDiffusion

        model = FlashDiffusion("runwayml/stable-diffusion-v1-5", device="cuda")
        images = model.generate("a castle on a hill, oil painting", seed=42)
        images[0].save("castle.png")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        scheduler: Optional[str] = None,
        safety_checker: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        self._pipe = None
        self._scheduler_name = scheduler
        self._safety_checker = safety_checker

    @property
    def pipe(self):
        """Lazy-load the diffusers pipeline."""
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def _load_pipeline(self):
        from diffusers import StableDiffusionPipeline, DiffusionPipeline

        logger.info("Loading pipeline: %s", self.model_id)

        kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
        }

        if not self._safety_checker:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        try:
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        except Exception:
            pipe = DiffusionPipeline.from_pretrained(self.model_id, **kwargs)

        if self._scheduler_name:
            pipe.scheduler = self._get_scheduler(pipe.scheduler.config)

        pipe = pipe.to(self.device)
        logger.info("Pipeline loaded on %s", self.device)
        return pipe

    def _get_scheduler(self, config):
        from flashdiffusion.schedulers import SCHEDULER_MAP

        name = self._scheduler_name.lower()
        if name not in SCHEDULER_MAP:
            logger.warning("Unknown scheduler '%s', using pipeline default", name)
            return None

        scheduler_cls = SCHEDULER_MAP[name]
        return scheduler_cls.from_config(config)

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
            prompt: Text prompt(s) for generation.
            negative_prompt: Negative prompt(s).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output image width.
            height: Output image height.
            seed: Random seed for reproducibility.
            num_images: Number of images to generate.

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

    def get_components(self) -> Dict[str, object]:
        """Return individual pipeline components."""
        pipe = self.pipe
        return {
            "unet": getattr(pipe, "unet", None),
            "vae": getattr(pipe, "vae", None),
            "text_encoder": getattr(pipe, "text_encoder", None),
            "tokenizer": getattr(pipe, "tokenizer", None),
            "scheduler": getattr(pipe, "scheduler", None),
        }

    def set_scheduler(self, scheduler_name: str):
        """Swap the noise scheduler at runtime."""
        self._scheduler_name = scheduler_name
        if self._pipe is not None:
            scheduler = self._get_scheduler(self._pipe.scheduler.config)
            if scheduler is not None:
                self._pipe.scheduler = scheduler

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt into embeddings."""
        pipe = self.pipe
        tokens = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return pipe.text_encoder(tokens)[0]

    def __repr__(self) -> str:
        return f"FlashDiffusion(model_id='{self.model_id}', device='{self.device}')"
