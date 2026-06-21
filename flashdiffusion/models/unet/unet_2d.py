"""UNet2D with cross-attention — wrapper around HuggingFace diffusers UNet."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UNet2DConditionWrapper(nn.Module):
    """Wrapper around diffusers UNet2DConditionModel.

    Provides a simplified interface for loading, forward pass, and
    component access while preserving full diffusers compatibility.

    Args:
        model_id: HuggingFace model ID to load the UNet from.
        subfolder: Subfolder within the model repo (default: "unet").
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "unet",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.subfolder = subfolder
        self.torch_dtype = torch_dtype or torch.float32
        self._unet = None

    @property
    def unet(self):
        """Lazy-load the UNet model."""
        if self._unet is None:
            self._unet = self._load_unet()
        return self._unet

    def _load_unet(self):
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder=self.subfolder,
            torch_dtype=self.torch_dtype,
        )
        logger.info("UNet loaded from %s/%s", self.model_id, self.subfolder)
        return unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            sample: Noisy latent tensor (B, C, H, W).
            timestep: Current diffusion timestep.
            encoder_hidden_states: Text encoder output (B, seq_len, dim).

        Returns:
            Predicted noise tensor.
        """
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return output.sample

    def get_attention_layers(self) -> Dict[str, nn.Module]:
        """Return all cross-attention layers for LoRA injection."""
        attention_layers = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                attention_layers[name] = module
        return attention_layers

    @property
    def config(self):
        return self.unet.config

    def parameters(self, recurse=True):
        return self.unet.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.unet.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.unet.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.unet.load_state_dict(*args, **kwargs)
