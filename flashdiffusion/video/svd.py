"""Stable Video Diffusion (SVD) — image-to-video generation.

Implements the SVD pipeline with temporal UNet and micro-conditioning
for generating short video clips from a single input image.

Reference: https://arxiv.org/abs/2311.15127
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class TemporalUNetWrapper(nn.Module):
    """Wrapper for SVD's temporal UNet with 3D convolutions and temporal attention.

    Args:
        model_id: SVD model ID.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid",
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
        from diffusers import UNetSpatioTemporalConditionModel

        self._unet = UNetSpatioTemporalConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=self.torch_dtype,
        )
        logger.info("SVD temporal UNet loaded from %s", self.model_id)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the temporal UNet.

        Args:
            sample: Noisy video latents (B, T, C, H, W).
            timestep: Diffusion timestep.
            encoder_hidden_states: Image encoder features.
            added_time_ids: Micro-conditioning time IDs (fps, motion, noise_aug).

        Returns:
            Predicted noise tensor.
        """
        output = self.unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            **kwargs,
        )
        return output.sample


class SVDPipeline:
    """Stable Video Diffusion image-to-video pipeline.

    Generates short video clips from a single conditioning image
    with configurable frame count, FPS, and motion intensity.

    Args:
        model_id: SVD model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
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
            from diffusers import StableVideoDiffusionPipeline

            self._pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype,
            ).to(self.device)
            logger.info("SVD pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        image: Union[str, Image.Image],
        num_frames: int = 25,
        num_inference_steps: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        width: int = 1024,
        height: int = 576,
        seed: Optional[int] = None,
        decode_chunk_size: int = 8,
    ) -> List[Image.Image]:
        """Generate video frames from a conditioning image.

        Args:
            image: Input conditioning image (path or PIL Image).
            num_frames: Number of frames to generate.
            num_inference_steps: Denoising steps.
            fps: Target frames per second (affects temporal conditioning).
            motion_bucket_id: Motion intensity (higher = more motion).
            noise_aug_strength: Noise augmentation for input image.
            width: Output width.
            height: Output height.
            seed: Random seed.
            decode_chunk_size: Chunk size for VAE decoding.

        Returns:
            List of PIL Image frames.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = image.resize((width, height), Image.LANCZOS)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        frames = self.pipe(
            image=image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
        ).frames[0]

        return frames

    def save_as_gif(self, frames: List[Image.Image], output_path: str, fps: int = 7):
        """Save video frames as an animated GIF."""
        if frames:
            frames[0].save(
                output_path, save_all=True, append_images=frames[1:],
                duration=1000 // fps, loop=0,
            )
            logger.info("SVD GIF saved: %s (%d frames)", output_path, len(frames))

    def save_as_mp4(self, frames: List[Image.Image], output_path: str, fps: int = 7):
        """Save video frames as MP4 (requires imageio)."""
        try:
            import imageio
            import numpy as np

            writer = imageio.get_writer(output_path, fps=fps)
            for frame in frames:
                writer.append_data(np.array(frame))
            writer.close()
            logger.info("SVD MP4 saved: %s (%d frames)", output_path, len(frames))
        except ImportError:
            logger.warning("MP4 export requires: pip install imageio[ffmpeg]")
