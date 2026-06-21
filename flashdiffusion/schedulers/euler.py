"""Euler and Euler Ancestral schedulers — fast single-step ODE solvers."""

from typing import Optional

import torch

from flashdiffusion.schedulers.base import BaseScheduler


class EulerScheduler(BaseScheduler):
    """Euler discrete scheduler — fast deterministic sampling."""

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = (torch.arange(num_inference_steps, device=device).flip(0) * step_ratio).long()
        self.num_inference_steps = num_inference_steps

        sigmas = []
        for t in self.timesteps:
            alpha_prod = self.alphas_cumprod[t]
            sigmas.append(((1 - alpha_prod) / alpha_prod) ** 0.5)
        sigmas.append(0.0)
        self.sigmas = torch.tensor(sigmas, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(step_idx) == 0:
            step_idx = 0
        else:
            step_idx = step_idx.item()

        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]

        sample - sigma * model_output
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output

        return prev_sample

    @classmethod
    def from_config(cls, config):
        return cls(
            num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
            beta_start=getattr(config, "beta_start", 0.00085),
            beta_end=getattr(config, "beta_end", 0.012),
            beta_schedule=getattr(config, "beta_schedule", "scaled_linear"),
        )


class EulerAncestralScheduler(BaseScheduler):
    """Euler Ancestral scheduler — stochastic variant with noise injection."""

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

        sigma_t = ((1 - alpha_prod_t) / alpha_prod_t) ** 0.5
        sigma_t_prev = ((1 - alpha_prod_t_prev) / alpha_prod_t_prev) ** 0.5

        sigma_up = (sigma_t_prev**2 * (sigma_t**2 - sigma_t_prev**2) / sigma_t**2).clamp(min=0) ** 0.5
        sigma_down = (sigma_t_prev**2 - sigma_up**2).clamp(min=0) ** 0.5

        prev_sample = alpha_prod_t_prev ** 0.5 * pred_x0 + sigma_down * model_output

        if t > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + sigma_up * noise

        return prev_sample

    @classmethod
    def from_config(cls, config):
        return cls(
            num_train_timesteps=getattr(config, "num_train_timesteps", 1000),
            beta_start=getattr(config, "beta_start", 0.00085),
            beta_end=getattr(config, "beta_end", 0.012),
            beta_schedule=getattr(config, "beta_schedule", "scaled_linear"),
        )
