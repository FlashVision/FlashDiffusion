"""Neural network building blocks for diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv2d + BatchNorm + Activation block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = "silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class AttentionBlock(nn.Module):
    """Self-attention block with layer normalization."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x + residual


class ResBlock(nn.Module):
    """Residual block with two convolutions and optional time embedding."""

    def __init__(self, channels: int, out_channels: int = None, time_emb_dim: int = None):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.skip = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and time_emb is not None:
            h = h + self.time_proj(self.act(time_emb))[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


__all__ = ["ConvModule", "AttentionBlock", "ResBlock"]
