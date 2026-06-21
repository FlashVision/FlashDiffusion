"""ESRGAN — Enhanced Super-Resolution GAN with RRDB architecture.

Implements the Residual-in-Residual Dense Block (RRDB) architecture
for 4x image super-resolution.

Reference: https://arxiv.org/abs/1809.00219
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class DenseBlock(nn.Module):
    """Dense block with 5 convolutional layers and growth rate.

    Each layer receives concatenated features from all previous layers.

    Args:
        channels: Number of input channels.
        growth_rate: Growth rate (channels added per layer).
    """

    def __init__(self, channels: int = 64, growth_rate: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block.

    Cascades three dense blocks with residual scaling.

    Args:
        channels: Number of feature channels.
        growth_rate: Dense block growth rate.
    """

    def __init__(self, channels: int = 64, growth_rate: int = 32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth_rate)
        self.rdb2 = DenseBlock(channels, growth_rate)
        self.rdb3 = DenseBlock(channels, growth_rate)
        self.beta = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x


class RRDBNet(nn.Module):
    """RRDB-based generator network for 4x super-resolution.

    Args:
        in_channels: Input image channels (3 for RGB).
        out_channels: Output image channels.
        num_features: Base feature channels.
        num_blocks: Number of RRDB blocks.
        growth_rate: Dense block growth rate.
        scale: Upscaling factor (2 or 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        growth_rate: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        self.body = nn.Sequential(*[
            RRDB(num_features, growth_rate) for _ in range(num_blocks)
        ])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.upsampler = nn.ModuleList()
        num_upsample = 0
        s = scale
        while s > 1:
            self.upsampler.append(nn.Conv2d(num_features, num_features * 4, 3, 1, 1))
            self.upsampler.append(nn.PixelShuffle(2))
            s //= 2
            num_upsample += 1

        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an input image.

        Args:
            x: Low-resolution input (B, 3, H, W) in [0, 1].

        Returns:
            Super-resolved output (B, 3, H*scale, W*scale) in [0, 1].
        """
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        for layer in self.upsampler:
            feat = layer(feat)
            if isinstance(layer, nn.Conv2d):
                feat = self.lrelu(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ESRGANUpscaler:
    """ESRGAN super-resolution upscaler.

    Args:
        model_path: Path to pre-trained ESRGAN weights.
        scale: Upscaling factor.
        device: Target device.
        tile_size: Tile size for processing large images (0 = no tiling).
        tile_overlap: Overlap between tiles.
    """

    PRETRAINED_MODELS = {
        "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "realesrgan-x4plus-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        scale: int = 4,
        device: str = "cuda",
        tile_size: int = 0,
        tile_overlap: int = 32,
    ):
        self.scale = scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.model = RRDBNet(scale=scale)

        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("ESRGAN weights loaded from %s", model_path)

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def upscale(self, image: Union[str, Image.Image]) -> Image.Image:
        """Upscale an image by the model's scale factor.

        Args:
            image: Input image (path or PIL Image).

        Returns:
            Upscaled PIL Image.
        """
        import numpy as np

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.tile_size > 0:
            output = self._tiled_inference(img_tensor)
        else:
            output = self.model(img_tensor)

        output = output.squeeze(0).clamp(0, 1).cpu()
        output_array = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(output_array)

    def _tiled_inference(self, img: torch.Tensor) -> torch.Tensor:
        """Process large images in overlapping tiles."""
        _, _, h, w = img.shape
        tile = self.tile_size
        overlap = self.tile_overlap
        output = torch.zeros(1, 3, h * self.scale, w * self.scale, device=self.device)
        weight = torch.zeros_like(output)

        for y in range(0, h, tile - overlap):
            for x in range(0, w, tile - overlap):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                tile_input = img[:, :, y:y_end, x:x_end]

                tile_output = self.model(tile_input)

                oy = y * self.scale
                ox = x * self.scale
                oh = (y_end - y) * self.scale
                ow = (x_end - x) * self.scale

                output[:, :, oy:oy + oh, ox:ox + ow] += tile_output
                weight[:, :, oy:oy + oh, ox:ox + ow] += 1

        return output / weight.clamp(min=1)

    def __call__(self, image: Union[str, Image.Image]) -> Image.Image:
        return self.upscale(image)
