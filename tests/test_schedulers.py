"""Unit tests for FlashDiffusion schedulers."""

import torch
import pytest
from flashdiffusion.schedulers import (
    DDPMScheduler, DDIMScheduler, DPMPlusPlusScheduler,
    EulerScheduler, EulerAncestralScheduler, LCMScheduler,
)


@pytest.mark.parametrize("SchedulerCls", [
    DDPMScheduler, DDIMScheduler, DPMPlusPlusScheduler,
    EulerScheduler, EulerAncestralScheduler, LCMScheduler,
])
def test_scheduler_set_timesteps(SchedulerCls):
    """All schedulers should produce valid timesteps after set_timesteps."""
    scheduler = SchedulerCls(num_train_timesteps=1000)
    scheduler.set_timesteps(20)
    assert len(scheduler.timesteps) == 20


@pytest.mark.parametrize("SchedulerCls", [
    DDIMScheduler, DPMPlusPlusScheduler, LCMScheduler,
])
def test_scheduler_step_output_shape(SchedulerCls):
    """Scheduler step should return tensor with same shape as input."""
    scheduler = SchedulerCls(num_train_timesteps=1000)
    scheduler.set_timesteps(20)

    sample = torch.randn(1, 4, 64, 64)
    model_output = torch.randn(1, 4, 64, 64)
    t = scheduler.timesteps[0].item()

    output = scheduler.step(model_output, t, sample)
    assert output.shape == sample.shape


def test_add_noise():
    """add_noise should produce a noisy version of the original sample."""
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    original = torch.ones(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    timesteps = torch.tensor([500])

    noisy = scheduler.add_noise(original, noise, timesteps)
    assert noisy.shape == original.shape
    assert not torch.allclose(noisy, original)


@pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "cosine"])
def test_beta_schedules(schedule):
    """All beta schedules should produce valid beta values."""
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule=schedule)
    assert (scheduler.betas > 0).all()
    assert (scheduler.betas < 1).all()
