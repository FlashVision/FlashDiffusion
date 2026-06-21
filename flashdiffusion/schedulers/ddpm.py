"""DDPM — Denoising Diffusion Probabilistic Models scheduler (Ho et al., 2020)."""

from typing import Optional

import torch

from flashdiffusion.schedulers.base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    """Original DDPM noise scheduler.

    Uses the full Markov chain with stochastic reverse steps.
    Requires many steps (50-1000) for high quality but is the reference
    implementation for training.
    """

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps, device=device).flip(0) * step_ratio
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
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_t = self.betas[t]

        pred_original = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t**0.5

        pred_original = torch.clamp(pred_original, -1, 1)

        (alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_prod_t)
        ((1 - alpha_prod_t_prev) ** 0.5 * self.alphas[t] ** 0.5) / (1 - alpha_prod_t) * (1 - alpha_prod_t_prev) / (
            1 - alpha_prod_t
        ) if t > 0 else 0

        mean = (alpha_prod_t_prev**0.5 * beta_t / (1 - alpha_prod_t)) * pred_original + (
            self.alphas[t] ** 0.5 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        ) * sample

        if t > 0:
            variance = beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            return mean + variance**0.5 * noise

        return mean
