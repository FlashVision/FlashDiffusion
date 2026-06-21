"""Diffusion training losses — MSE noise prediction and v-prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def noise_prediction_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard MSE noise prediction loss for diffusion training.

    Args:
        noise_pred: Predicted noise from UNet.
        noise_target: Ground truth noise.
        reduction: Loss reduction mode ("mean", "sum", "none").

    Returns:
        MSE loss value.
    """
    return F.mse_loss(noise_pred, noise_target, reduction=reduction)


def v_prediction_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """V-prediction loss for velocity parameterization.

    Used by some schedulers (e.g. in SDXL) where the model predicts
    v = alpha_t * noise - sigma_t * x_0 instead of noise directly.

    Args:
        v_pred: Predicted velocity.
        v_target: Target velocity.
        reduction: Loss reduction mode.

    Returns:
        MSE loss value.
    """
    return F.mse_loss(v_pred, v_target, reduction=reduction)


class DiffusionLoss(nn.Module):
    """Configurable diffusion training loss.

    Supports both noise prediction (epsilon) and velocity (v) prediction
    parameterizations with optional SNR weighting.

    Args:
        prediction_type: "epsilon" for noise prediction, "v_prediction" for velocity.
        snr_gamma: If set, apply Min-SNR weighting (Hang et al., 2023).
    """

    def __init__(
        self,
        prediction_type: str = "epsilon",
        snr_gamma: float = 0.0,
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor = None,
        alphas_cumprod: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.prediction_type == "v_prediction":
            loss = v_prediction_loss(model_output, target, reduction="none")
        else:
            loss = noise_prediction_loss(model_output, target, reduction="none")

        loss = loss.mean(dim=list(range(1, loss.ndim)))

        if self.snr_gamma > 0 and timesteps is not None and alphas_cumprod is not None:
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            weight = torch.clamp(snr, max=self.snr_gamma) / snr
            loss = loss * weight

        return loss.mean()
