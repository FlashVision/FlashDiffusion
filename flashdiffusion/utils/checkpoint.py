"""Checkpoint utilities for saving and loading models."""

import os
from typing import Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    save_path: str,
    config: Optional[Dict] = None,
    **extra,
) -> str:
    """Save training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        step: Current training step.
        loss: Current loss value.
        save_path: Destination path.
        config: Configuration dict (optional).
        **extra: Additional entries to save.

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **extra,
    }

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
    return save_path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict:
    """Load training checkpoint.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint.
        optimizer: Optimizer to load state into (optional).
        device: Device to load to.

    Returns:
        Checkpoint dictionary with step, loss, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"Model loaded from: {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded")
        except (ValueError, KeyError) as e:
            print(f"  Optimizer state skipped ({e}). Starting fresh.")

    return {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
    }
