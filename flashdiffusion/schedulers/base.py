"""Base scheduler class — defines the interface for all noise schedulers."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseScheduler(ABC):
    """Abstract base class for diffusion noise schedulers.

    All schedulers must implement:
        - ``set_timesteps(num_steps)``: configure the denoising schedule
        - ``step(model_output, timestep, sample)``: perform one denoising step
        - ``add_noise(original, noise, timesteps)``: add noise to clean samples

    Args:
        num_train_timesteps: Total number of diffusion timesteps used during training.
        beta_start: Starting beta value for the noise schedule.
        beta_end: Ending beta value for the noise schedule.
        beta_schedule: Type of schedule ("linear", "scaled_linear", "cosine").
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def _compute_betas(self) -> torch.Tensor:
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            return torch.linspace(
                self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps
            ) ** 2
        elif self.beta_schedule == "cosine":
            steps = self.num_train_timesteps + 1
            t = torch.linspace(0, 1, steps)
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set the timestep schedule for inference."""

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Perform one denoising step."""

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean samples according to the noise schedule."""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5

        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    @classmethod
    def from_config(cls, config):
        """Create scheduler from a diffusers config dict."""
        return cls(
            num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
            beta_start=getattr(config, "beta_start", 0.00085),
            beta_end=getattr(config, "beta_end", 0.012),
            beta_schedule=getattr(config, "beta_schedule", "scaled_linear"),
        )
