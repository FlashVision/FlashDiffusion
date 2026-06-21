"""LoRA (Low-Rank Adaptation) for diffusion model fine-tuning.

Supports applying LoRA to UNet cross-attention layers and text encoders
using the HuggingFace PEFT / diffusers LoRA infrastructure.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_lora(
    unet: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to UNet cross-attention layers.

    Freezes all base weights and injects low-rank trainable matrices into
    the attention projection layers (to_q, to_k, to_v, to_out).

    Args:
        unet: UNet2DConditionModel from diffusers.
        rank: LoRA rank (lower = fewer params, higher = more capacity).
        alpha: Scaling factor (alpha/rank is the effective scale).
        target_modules: List of module name patterns to apply LoRA to.

    Returns:
        UNet with LoRA applied. Only LoRA parameters are trainable.
    """
    if target_modules is None:
        target_modules = ["to_q", "to_v", "to_k", "to_out.0"]

    for param in unet.parameters():
        param.requires_grad = False

    replaced = []
    _inject_lora_layers(unet, rank, alpha, target_modules, replaced, prefix="")

    total = sum(p.numel() for p in unet.parameters())
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied: %d layers adapted (rank=%d, alpha=%.1f). Trainable: %d / %d params (%.1f%%)",
        len(replaced),
        rank,
        alpha,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    return unet


def _inject_lora_layers(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_modules: List[str],
    replaced: list,
    prefix: str,
):
    """Recursively inject LoRA into matching Linear layers."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and any(t in full_name for t in target_modules):
            lora_layer = LoRALinear(
                child.in_features,
                child.out_features,
                rank=rank,
                alpha=alpha,
                bias=child.bias is not None,
            )
            lora_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_layer)
            replaced.append(full_name)
        else:
            _inject_lora_layers(child, rank, alpha, target_modules, replaced, full_name)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    output = W_frozen @ x + (alpha/rank) * B @ A @ x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base weights for zero-overhead inference.

    After merging, LoRA layers become standard Linear layers with
    W_merged = W + (alpha/rank) * B @ A.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.weight.data += delta
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            merged_count += 1

    logger.info("Merged LoRA weights in %d layers", merged_count)
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA adapter weights for saving."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_weights(model: nn.Module, lora_path: str, device: str = "cpu"):
    """Load LoRA adapter weights into a model with LoRA layers."""
    state = torch.load(lora_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    logger.info("LoRA weights loaded from %s", lora_path)
