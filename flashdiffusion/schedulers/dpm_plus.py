"""DPM++ — DPM++ 2M and DPM++ 2M SDE schedulers (Lu et al., 2022)."""

from typing import Optional

import torch

from flashdiffusion.schedulers.base import BaseScheduler


class DPMPlusPlusScheduler(BaseScheduler):
    """DPM++ 2M scheduler for high-quality generation in 15-30 steps."""

    def __init__(self, use_sde: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_sde = use_sde

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = (torch.arange(num_inference_steps, device=device).flip(0) * step_ratio).long()
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

        prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0 + \
                      (1 - alpha_prod_t_prev) ** 0.5 * model_output

        if self.use_sde and t > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            sigma = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) ** 0.5
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
