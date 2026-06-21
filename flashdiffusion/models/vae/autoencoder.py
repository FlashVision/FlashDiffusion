"""VAE encoder/decoder — wrapper around HuggingFace diffusers AutoencoderKL."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AutoencoderWrapper(nn.Module):
    """Wrapper around diffusers AutoencoderKL.

    Provides encode/decode interface for converting between pixel space
    and latent space used by diffusion models.

    Args:
        model_id: HuggingFace model ID.
        subfolder: Subfolder within the model repo (default: "vae").
        torch_dtype: Data type for model weights.
    """

    LATENT_SCALE_FACTOR = 0.18215

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "vae",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.subfolder = subfolder
        self.torch_dtype = torch_dtype or torch.float32
        self._vae = None

    @property
    def vae(self):
        """Lazy-load the VAE model."""
        if self._vae is None:
            self._vae = self._load_vae()
        return self._vae

    def _load_vae(self):
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder=self.subfolder,
            torch_dtype=self.torch_dtype,
        )
        logger.info("VAE loaded from %s/%s", self.model_id, self.subfolder)
        return vae

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to latent space.

        Args:
            pixel_values: Image tensor (B, 3, H, W) in [-1, 1].

        Returns:
            Latent tensor (B, 4, H/8, W/8).
        """
        posterior = self.vae.encode(pixel_values).latent_dist
        latents = posterior.sample() * self.LATENT_SCALE_FACTOR
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent-space tensors to pixel-space images.

        Args:
            latents: Latent tensor (B, 4, H/8, W/8).

        Returns:
            Image tensor (B, 3, H, W) in [-1, 1].
        """
        latents = latents / self.LATENT_SCALE_FACTOR
        image = self.vae.decode(latents).sample
        return image

    def forward(self, x: torch.Tensor, encode: bool = True) -> torch.Tensor:
        if encode:
            return self.encode(x)
        return self.decode(x)
