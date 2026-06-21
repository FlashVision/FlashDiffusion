"""ControlNet conditioning — wrapper for HuggingFace diffusers ControlNetModel."""

import logging
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

CONTROLNET_MODELS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "lllyasviel/sd-controlnet-depth",
    "pose": "lllyasviel/sd-controlnet-openpose",
    "scribble": "lllyasviel/sd-controlnet-scribble",
    "hed": "lllyasviel/sd-controlnet-hed",
    "mlsd": "lllyasviel/sd-controlnet-mlsd",
    "seg": "lllyasviel/sd-controlnet-seg",
    "normal": "lllyasviel/sd-controlnet-normal",
}

SDXL_CONTROLNET_MODELS = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",
    "scribble": "xinsir/controlnet-scribble-sdxl-1.0",
}


class ControlNetWrapper:
    """Wrapper for ControlNet models and condition preprocessing.

    Args:
        controlnet_type: Type of ControlNet ("canny", "depth", "pose", etc.)
            or a HuggingFace model ID.
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        controlnet_type: str = "canny",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.controlnet_type = controlnet_type
        self.torch_dtype = torch_dtype or torch.float16
        self.model_id = CONTROLNET_MODELS.get(controlnet_type, controlnet_type)
        self._model = None

    @property
    def model(self):
        """Lazy-load the ControlNet model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        from diffusers import ControlNetModel

        model = ControlNetModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )
        logger.info("ControlNet loaded: %s (%s)", self.controlnet_type, self.model_id)
        return model

    def preprocess(
        self,
        image: Union[str, Image.Image],
        resolution: int = 512,
    ) -> Image.Image:
        """Preprocess an image to create a ControlNet condition map.

        Args:
            image: Input image path or PIL Image.
            resolution: Target resolution.

        Returns:
            Preprocessed condition image.
        """
        import numpy as np
        import cv2

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = image.resize((resolution, resolution), Image.BILINEAR)
        img_array = np.array(image)

        if self.controlnet_type == "canny":
            edges = cv2.Canny(img_array, 100, 200)
            condition = Image.fromarray(edges).convert("RGB")
        elif self.controlnet_type == "depth":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            condition = Image.fromarray(gray).convert("RGB")
        else:
            condition = image

        return condition

    def get_pipeline_kwargs(self) -> Dict:
        """Return kwargs for constructing a ControlNet pipeline."""
        return {"controlnet": self.model}


class SDXLControlNetWrapper:
    """ControlNet wrapper extended for SDXL architecture.

    Loads SDXL-compatible ControlNet models and provides conditioning
    for the SDXL UNet with dual text encoder support.

    Args:
        controlnet_type: Type of ControlNet or HuggingFace model ID.
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        controlnet_type: str = "canny",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.controlnet_type = controlnet_type
        self.torch_dtype = torch_dtype or torch.float16
        self.model_id = SDXL_CONTROLNET_MODELS.get(controlnet_type, controlnet_type)
        self._model = None
        self._pipe = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        from diffusers import ControlNetModel

        model = ControlNetModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )
        logger.info("SDXL ControlNet loaded: %s (%s)", self.controlnet_type, self.model_id)
        return model

    def preprocess(
        self,
        image: Union[str, Image.Image],
        resolution: int = 1024,
    ) -> Image.Image:
        """Preprocess image for SDXL ControlNet conditioning.

        Args:
            image: Input image.
            resolution: Target resolution (default 1024 for SDXL).

        Returns:
            Preprocessed condition image.
        """
        import numpy as np
        import cv2

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = image.resize((resolution, resolution), Image.BILINEAR)
        img_array = np.array(image)

        if self.controlnet_type == "canny":
            edges = cv2.Canny(img_array, 100, 200)
            condition = Image.fromarray(edges).convert("RGB")
        elif self.controlnet_type == "depth":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            condition = Image.fromarray(gray).convert("RGB")
        else:
            condition = image

        return condition

    def create_pipeline(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
    ):
        """Create an SDXL ControlNet pipeline.

        Args:
            base_model_id: SDXL base model ID.
            device: Target device.

        Returns:
            Configured StableDiffusionXLControlNetPipeline.
        """
        from diffusers import StableDiffusionXLControlNetPipeline

        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.model,
            torch_dtype=self.torch_dtype,
        ).to(device)
        logger.info("SDXL ControlNet pipeline created with %s", base_model_id)
        return self._pipe

    def generate(
        self,
        prompt: Union[str, List[str]],
        control_image: Union[str, Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate SDXL images with ControlNet conditioning.

        Args:
            prompt: Text prompt.
            control_image: ControlNet condition image.
            negative_prompt: Negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            controlnet_conditioning_scale: ControlNet strength (0-1).
            width: Output width.
            height: Output height.
            seed: Random seed.

        Returns:
            List of generated PIL Images.
        """
        if self._pipe is None:
            self.create_pipeline()

        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")

        control_image = control_image.resize((width, height), Image.BILINEAR)

        device = self._pipe.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width, height=height,
            generator=generator,
        )
        return result.images

    def get_pipeline_kwargs(self) -> Dict:
        return {"controlnet": self.model}
