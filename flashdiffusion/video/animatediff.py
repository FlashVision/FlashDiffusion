"""AnimateDiff — Motion module for video generation from SD image models.

Inserts temporal attention layers into the UNet to generate consistent
video frames from a text-to-image model.

Reference: https://arxiv.org/abs/2307.04725
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Temporal self-attention layer for cross-frame consistency.

    Applied along the temporal dimension to establish relationships
    between video frames.

    Args:
        channels: Feature channel dimension.
        num_heads: Number of attention heads.
        num_frames: Number of video frames.
    """

    def __init__(self, channels: int, num_heads: int = 8, num_frames: int = 16):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(0.0),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)

    def forward(self, x: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
        """Apply temporal attention across frames.

        Args:
            x: Feature tensor (B*T, C, H, W).
            num_frames: Number of frames T.

        Returns:
            Temporally attended features (B*T, C, H, W).
        """
        bt, c, h, w = x.shape
        batch_size = bt // num_frames

        residual = x
        x = self.norm(x)

        x = x.reshape(batch_size, num_frames, c, h * w)
        x = x.permute(0, 3, 1, 2).reshape(batch_size * h * w, num_frames, c)

        pos = self.pos_embedding[:, :num_frames, :]
        x = x + pos

        qkv = self.to_qkv(x).reshape(
            batch_size * h * w, num_frames, 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(batch_size * h * w, num_frames, c)
        out = self.to_out(out)

        out = out.reshape(batch_size, h * w, num_frames, c)
        out = out.permute(0, 2, 3, 1).reshape(bt, c, h, w)

        return out + residual


class AnimateDiffMotionModule(nn.Module):
    """AnimateDiff motion module — collection of temporal attention layers.

    Designed to be inserted into an existing SD UNet to enable
    video generation without retraining the spatial layers.

    Args:
        channels_list: List of channel dimensions for each UNet level.
        num_frames: Number of video frames.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        channels_list: Optional[List[int]] = None,
        num_frames: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        if channels_list is None:
            channels_list = [320, 640, 1280, 1280]

        self.num_frames = num_frames
        self.temporal_attentions = nn.ModuleDict()

        for i, channels in enumerate(channels_list):
            self.temporal_attentions[f"down_{i}"] = TemporalAttention(
                channels, num_heads=num_heads, num_frames=num_frames,
            )
            self.temporal_attentions[f"up_{i}"] = TemporalAttention(
                channels, num_heads=num_heads, num_frames=num_frames,
            )

        self.temporal_attentions["mid"] = TemporalAttention(
            channels_list[-1], num_heads=num_heads, num_frames=num_frames,
        )

    def apply_temporal_attention(
        self,
        x: torch.Tensor,
        block_name: str,
    ) -> torch.Tensor:
        """Apply temporal attention for a specific UNet block.

        Args:
            x: Feature tensor (B*T, C, H, W).
            block_name: Block identifier (e.g., "down_0", "mid", "up_3").

        Returns:
            Temporally processed features.
        """
        if block_name in self.temporal_attentions:
            return self.temporal_attentions[block_name](x, self.num_frames)
        return x


class AnimateDiffPipeline:
    """AnimateDiff video generation pipeline.

    Combines a Stable Diffusion model with AnimateDiff motion modules
    to generate short video clips from text prompts.

    Args:
        model_id: Base SD model ID.
        motion_adapter_id: AnimateDiff motion adapter ID.
        device: Target device.
        torch_dtype: Data type.
        num_frames: Number of frames to generate.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        num_frames: int = 16,
    ):
        self.model_id = model_id
        self.motion_adapter_id = motion_adapter_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.num_frames = num_frames
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from diffusers import AnimateDiffPipeline as _AnimPipe, MotionAdapter, DDIMScheduler

            adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_id, torch_dtype=self.torch_dtype,
            )
            self._pipe = _AnimPipe.from_pretrained(
                self.model_id, motion_adapter=adapter,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self._pipe.scheduler = DDIMScheduler.from_config(self._pipe.scheduler.config)
            logger.info("AnimateDiff pipeline loaded: %s + %s", self.model_id, self.motion_adapter_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_frames: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate video frames from a text prompt.

        Args:
            prompt: Text prompt.
            negative_prompt: Negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            num_frames: Number of frames (overrides init value).
            width: Frame width.
            height: Frame height.
            seed: Random seed.

        Returns:
            List of PIL Image frames.
        """
        num_frames = num_frames or self.num_frames

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width, height=height,
            generator=generator,
        )
        return result.frames[0]

    def save_as_gif(self, frames: List[Image.Image], output_path: str, fps: int = 8):
        """Save generated frames as an animated GIF."""
        if frames:
            frames[0].save(
                output_path, save_all=True, append_images=frames[1:],
                duration=1000 // fps, loop=0,
            )
            logger.info("Saved GIF: %s (%d frames, %d fps)", output_path, len(frames), fps)
