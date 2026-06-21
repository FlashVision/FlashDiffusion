"""SDXL UNet wrapper with attention pooling and additional conditioning."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SDXLUNetWrapper(nn.Module):
    """Wrapper for SDXL's UNet2DConditionModel.

    Extends the base UNet with SDXL-specific features:
    - Dual cross-attention for concatenated CLIP embeddings
    - Additional conditioning via add_text_embeds and add_time_ids
    - Attention pooling from the second text encoder

    Args:
        model_id: SDXL model ID.
        torch_dtype: Data type for weights.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.torch_dtype = torch_dtype or torch.float16
        self._unet = None

    @property
    def unet(self):
        if self._unet is None:
            self._load()
        return self._unet

    def _load(self):
        from diffusers import UNet2DConditionModel

        self._unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=self.torch_dtype,
        )
        logger.info("SDXL UNet loaded from %s", self.model_id)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through SDXL UNet.

        Args:
            sample: Noisy latent (B, 4, H/8, W/8).
            timestep: Diffusion timestep.
            encoder_hidden_states: Concatenated text embeddings (B, seq_len, 2048).
            added_cond_kwargs: Dict with "text_embeds" and "time_ids" for SDXL conditioning.

        Returns:
            Predicted noise tensor.
        """
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            **kwargs,
        )
        return output.sample

    @staticmethod
    def compute_time_ids(
        original_size: Tuple[int, int],
        crop_coords: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = (1024, 1024),
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Compute SDXL micro-conditioning time IDs.

        Args:
            original_size: Original image dimensions (h, w).
            crop_coords: Top-left crop coordinates.
            target_size: Target generation size.

        Returns:
            Time IDs tensor of shape (1, 6).
        """
        return torch.tensor(
            [original_size[0], original_size[1], crop_coords[0], crop_coords[1], target_size[0], target_size[1]],
            dtype=dtype,
            device=device,
        ).unsqueeze(0)

    def get_attention_layers(self) -> Dict[str, nn.Module]:
        layers = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                layers[name] = module
        return layers

    @property
    def config(self):
        return self.unet.config

    def parameters(self, recurse=True):
        return self.unet.parameters(recurse=recurse)
