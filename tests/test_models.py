"""Unit tests for FlashDiffusion models."""

import torch
import pytest
from flashdiffusion.models.lora import LoRALinear, apply_lora, merge_lora_weights


def test_lora_linear_forward():
    """LoRA linear layer produces output with correct shape."""
    layer = LoRALinear(in_features=64, out_features=128, rank=4, alpha=4.0)
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 128)


def test_lora_linear_nonzero_output():
    """LoRA linear should produce non-zero output."""
    layer = LoRALinear(in_features=32, out_features=32, rank=4, alpha=4.0)
    nn_mod = torch.nn.Linear(32, 32)
    layer.weight.data.copy_(nn_mod.weight.data)
    if nn_mod.bias is not None:
        layer.bias.data.copy_(nn_mod.bias.data)

    x = torch.randn(1, 32)
    out = layer(x)
    assert out.abs().sum() > 0


def test_lora_merge_resets_adapters():
    """After merging, LoRA A and B matrices should be zeroed."""
    layer = LoRALinear(in_features=32, out_features=32, rank=4, alpha=4.0)
    merge_lora_weights(layer)
    assert (layer.lora_A.abs().sum() + layer.lora_B.abs().sum()).item() == 0


def test_lora_rank_reduces_params():
    """Higher rank should have more trainable parameters."""
    layer_r2 = LoRALinear(64, 64, rank=2)
    layer_r8 = LoRALinear(64, 64, rank=8)

    params_r2 = sum(p.numel() for p in [layer_r2.lora_A, layer_r2.lora_B])
    params_r8 = sum(p.numel() for p in [layer_r8.lora_A, layer_r8.lora_B])

    assert params_r2 < params_r8
