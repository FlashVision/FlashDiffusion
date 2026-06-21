"""DDIM — Denoising Diffusion Implicit Models scheduler (Song et al., 2021)."""

from typing import Optional

import torch

from flashdiffusion.schedulers.base import BaseScheduler


class DDIMScheduler(BaseScheduler):
    """DDIM deterministic scheduler.

    Enables fast generation with fewer steps (20-50) while maintaining
    quality.  The deterministic formulation removes stochasticity, making
    outputs reproducible for a given seed.
    """

    def __init__(self, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps, device=device).flip(0) * step_ratio).long()
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]

        t_prev = max(t - self.num_train_timesteps // self.num_inference_steps, 0)
        alpha_prod_t_prev = self.alphas_cumprod[t_prev]

        pred_x0 = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        sigma = self.eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) ** 0.5

        pred_dir = (1 - alpha_prod_t_prev - sigma**2).clamp(min=0) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        if self.eta > 0 and t > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + sigma * noise

        return prev_sample

    @classmethod
    def from_config(cls, config):
        return cls(
            num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
            beta_start=getattr(config, "beta_start", 0.00085),
            beta_end=getattr(config, "beta_end", 0.012),
            beta_schedule=getattr(config, "beta_schedule", "scaled_linear"),
        )
