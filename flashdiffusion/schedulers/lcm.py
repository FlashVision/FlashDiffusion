"""LCM — Latent Consistency Model scheduler (Luo et al., 2023).

Enables high-quality image generation in 1-4 steps.
"""

from typing import Optional

import torch

from flashdiffusion.schedulers.base import BaseScheduler


class LCMScheduler(BaseScheduler):
    """Latent Consistency Model scheduler for ultra-fast generation."""

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps, device=device).flip(0) * step_ratio).long()
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]

        pred_x0 = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t**0.5
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        t_prev = max(t - self.num_train_timesteps // self.num_inference_steps, 0)

        if t_prev > 0:
            alpha_prod_t_prev = self.alphas_cumprod[t_prev]
            prev_sample = alpha_prod_t_prev**0.5 * pred_x0 + (1 - alpha_prod_t_prev) ** 0.5 * model_output
        else:
            prev_sample = pred_x0

        return prev_sample

    @classmethod
    def from_config(cls, config):
        return cls(
            num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
            beta_start=getattr(config, "beta_start", 0.00085),
            beta_end=getattr(config, "beta_end", 0.012),
            beta_schedule=getattr(config, "beta_schedule", "scaled_linear"),
        )
