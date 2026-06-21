"""IP-Adapter — Image Prompt Adapter for image-conditioned generation.

Uses a CLIP image encoder and decoupled cross-attention to condition
the diffusion model on reference images alongside text prompts.

Reference: https://arxiv.org/abs/2308.06721
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class IPAdapterImageEncoder(nn.Module):
    """CLIP image encoder for extracting visual features for IP-Adapter.

    Args:
        model_name: CLIP model name.
        projection_dim: Output projection dimension.
        num_tokens: Number of image tokens to produce.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        projection_dim: int = 768,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.num_tokens = num_tokens
        self._encoder = None
        self._processor = None

        self.projection = nn.Linear(1024, projection_dim * num_tokens)

    def _load(self):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        self._encoder = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
        self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
        logger.info("IP-Adapter image encoder loaded: %s", self.model_name)

    @property
    def encoder(self):
        if self._encoder is None:
            self._load()
        return self._encoder

    @property
    def processor(self):
        if self._processor is None:
            self._load()
        return self._processor

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode image(s) into IP-Adapter embeddings.

        Args:
            image: PIL Image(s).
            device: Target device.

        Returns:
            Image embeddings of shape (B, num_tokens, projection_dim).
        """
        if isinstance(image, Image.Image):
            image = [image]

        inputs = self.processor(images=image, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.encoder(**inputs)
        image_embeds = outputs.image_embeds

        projected = self.projection(image_embeds)
        return projected.reshape(-1, self.num_tokens, self.projection_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        projected = self.projection(outputs.image_embeds)
        return projected.reshape(-1, self.num_tokens, self.projection_dim)


class DecoupledCrossAttention(nn.Module):
    """Decoupled cross-attention for IP-Adapter.

    Adds a separate cross-attention branch for image tokens
    parallel to the text cross-attention, with a learnable scale.

    Args:
        hidden_size: Attention hidden dimension.
        cross_attention_dim: Conditioning dimension.
        num_heads: Number of attention heads.
        scale: Initial scale for image attention.
    """

    def __init__(
        self,
        hidden_size: int = 320,
        cross_attention_dim: int = 768,
        num_heads: int = 8,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale_param = nn.Parameter(torch.tensor(scale))

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention with image embeddings.

        Args:
            query: Query from self-attention (B, seq_len, hidden).
            image_embeds: IP-Adapter image embeddings (B, num_tokens, dim).

        Returns:
            Weighted image attention output.
        """
        batch, seq_len, hidden = query.shape

        k_ip = self.to_k_ip(image_embeds).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_ip = self.to_v_ip(image_embeds).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = query.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k_ip.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v_ip

        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return out * self.scale_param


class IPAdapter:
    """IP-Adapter for image-conditioned diffusion generation.

    Injects image features into the UNet's cross-attention layers
    using decoupled cross-attention.

    Args:
        pipe: Diffusion pipeline to augment.
        image_encoder_model: CLIP model name for image encoding.
        scale: IP-Adapter conditioning scale (0-1).
        num_tokens: Number of image tokens.
    """

    def __init__(
        self,
        pipe,
        image_encoder_model: str = "openai/clip-vit-large-patch14",
        scale: float = 0.6,
        num_tokens: int = 4,
    ):
        self.pipe = pipe
        self.scale = scale
        self.image_encoder = IPAdapterImageEncoder(
            model_name=image_encoder_model,
            num_tokens=num_tokens,
        )

    def generate(
        self,
        prompt: Union[str, List[str]],
        ip_adapter_image: Union[Image.Image, List[Image.Image]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images conditioned on both text and reference image.

        Args:
            prompt: Text prompt.
            ip_adapter_image: Reference image(s) for style/content conditioning.
            negative_prompt: Negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            width: Output width.
            height: Output height.
            seed: Random seed.

        Returns:
            List of generated PIL Images.
        """
        device = self.pipe.device if hasattr(self.pipe, "device") else torch.device("cuda")

        image_embeds = self.image_encoder.encode_image(ip_adapter_image, device=device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        result = self.pipe.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=[image_embeds],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width, height=height,
            generator=generator,
        )
        return result.images
