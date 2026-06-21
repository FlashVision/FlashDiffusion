"""SD3 / FLUX support — MMDiT, flow matching, and T5 text encoder.

Implements the Multimodal Diffusion Transformer (MMDiT) architecture
used in Stable Diffusion 3 and FLUX models with flow matching.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class MMDiTBlock(nn.Module):
    """Multimodal Diffusion Transformer (MMDiT) block.

    Processes both image and text tokens with separate streams
    that share attention weights.

    Args:
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.qkv_txt = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj_img = nn.Linear(hidden_size, hidden_size)
        self.proj_txt = nn.Linear(hidden_size, hidden_size)

        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with joint attention between image and text streams.

        Args:
            img_tokens: Image token features (B, N_img, hidden_size).
            txt_tokens: Text token features (B, N_txt, hidden_size).
            timestep_emb: Timestep embedding (B, hidden_size).

        Returns:
            Tuple of updated (img_tokens, txt_tokens).
        """
        mod = self.adaLN_modulation(timestep_emb).unsqueeze(1)
        shift_img, scale_img, gate_img, shift_txt, scale_txt, gate_txt = mod.chunk(6, dim=-1)

        img_norm = self.norm1_img(img_tokens) * (1 + scale_img) + shift_img
        txt_norm = self.norm1_txt(txt_tokens) * (1 + scale_txt) + shift_txt

        qkv_img = self.qkv_img(img_norm).reshape(
            img_norm.shape[0], img_norm.shape[1], 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        qkv_txt = self.qkv_txt(txt_norm).reshape(
            txt_norm.shape[0], txt_norm.shape[1], 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)

        q = torch.cat([qkv_img[0], qkv_txt[0]], dim=2)
        k = torch.cat([qkv_img[1], qkv_txt[1]], dim=2)
        v = torch.cat([qkv_img[2], qkv_txt[2]], dim=2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        n_img = img_norm.shape[1]
        img_attn = out[:, :, :n_img].transpose(1, 2).reshape(
            img_norm.shape[0], n_img, self.hidden_size,
        )
        txt_attn = out[:, :, n_img:].transpose(1, 2).reshape(
            txt_norm.shape[0], txt_norm.shape[1], self.hidden_size,
        )

        img_tokens = img_tokens + gate_img * self.proj_img(img_attn)
        txt_tokens = txt_tokens + gate_txt * self.proj_txt(txt_attn)

        img_tokens = img_tokens + gate_img * self.mlp_img(
            self.norm2_img(img_tokens) * (1 + scale_img) + shift_img
        )
        txt_tokens = txt_tokens + gate_txt * self.mlp_txt(
            self.norm2_txt(txt_tokens) * (1 + scale_txt) + shift_txt
        )

        return img_tokens, txt_tokens


class FlowMatchingScheduler:
    """Flow matching scheduler for SD3/FLUX.

    Implements the rectified flow ODE for continuous-time diffusion
    with a learned velocity field.

    Args:
        num_inference_steps: Default number of inference steps.
        shift: Timestep shift factor.
    """

    def __init__(self, num_inference_steps: int = 28, shift: float = 3.0):
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        self._timesteps = None

    def set_timesteps(self, num_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_steps
        sigmas = torch.linspace(1, 0, num_steps + 1, device=device)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self._timesteps = sigmas[:-1]
        self._sigmas = sigmas

    @property
    def timesteps(self) -> torch.Tensor:
        if self._timesteps is None:
            self.set_timesteps(self.num_inference_steps)
        return self._timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Perform a single flow matching step.

        Args:
            model_output: Predicted velocity.
            timestep: Current timestep.
            sample: Current noisy sample.
            step_index: Current step index.

        Returns:
            Denoised sample for next step.
        """
        sigma = self._sigmas[step_index]
        sigma_next = self._sigmas[step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        return prev_sample

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Add noise at a given sigma level (linear interpolation)."""
        return (1 - sigma) * original + sigma * noise


class SD3Pipeline:
    """Stable Diffusion 3 pipeline with MMDiT and flow matching.

    Args:
        model_id: SD3 model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
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
            from diffusers import StableDiffusion3Pipeline

            self._pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype,
            ).to(self.device)
            logger.info("SD3 pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width, height=height,
            generator=generator, num_images_per_prompt=num_images,
        )
        return result.images


class FLUXPipeline:
    """FLUX pipeline for high-quality image generation.

    Args:
        model_id: FLUX model ID.
        device: Target device.
        torch_dtype: Data type.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
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
            from diffusers import FluxPipeline

            self._pipe = FluxPipeline.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype,
            ).to(self.device)
            logger.info("FLUX pipeline loaded: %s", self.model_id)
        return self._pipe

    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width, height=height,
            generator=generator, num_images_per_prompt=num_images,
        )
        return result.images
