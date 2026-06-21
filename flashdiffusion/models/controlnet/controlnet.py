"""ControlNet conditioning — wrapper for HuggingFace diffusers ControlNetModel."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
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
