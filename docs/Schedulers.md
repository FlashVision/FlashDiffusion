# Schedulers

## Available Schedulers

| Scheduler | Steps | Quality | Speed | Best For |
|-----------|-------|---------|-------|----------|
| DDPM | 50-1000 | High | Slow | Training reference |
| DDIM | 20-50 | High | Fast | General purpose |
| DPM++ | 15-30 | Very High | Fast | Best quality/speed |
| Euler | 20-30 | High | Fast | Real-time generation |
| Euler Ancestral | 20-30 | High | Fast | Creative/stochastic |
| LCM | 1-4 | Good | Very Fast | Instant generation |

## Usage

```python
from flashdiffusion import FlashDiffusion

model = FlashDiffusion("runwayml/stable-diffusion-v1-5", scheduler="ddim")
images = model.generate("a cat", num_inference_steps=30)
```

## Custom Schedulers

```python
from flashdiffusion.schedulers import DDIMScheduler

scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(30)

for t in scheduler.timesteps:
    noise_pred = unet(latents, t, encoder_hidden_states)
    latents = scheduler.step(noise_pred, t, latents)
```

## API

All schedulers implement:
- `set_timesteps(num_steps)` — configure the denoising schedule
- `step(model_output, timestep, sample)` — perform one denoising step
- `add_noise(original, noise, timesteps)` — add noise to clean samples
