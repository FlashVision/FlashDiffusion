# Models

## Supported Models

| Model | Model ID | Parameters | VRAM | Speed (A100) |
|-------|----------|------------|------|--------------|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | 860M | ~4 GB | ~2.5s/img |
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | 3.5B | ~8 GB | ~5.0s/img |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | 12B | ~24 GB | ~12s/img |

## Architecture

- **UNet**: UNet2DConditionModel with cross-attention for noise prediction
- **VAE**: AutoencoderKL for latent space encoding/decoding
- **Text Encoder**: CLIP text encoder for prompt conditioning
- **Scheduler**: Configurable noise scheduler (DDIM, DPM++, Euler, LCM)

## Model Loading

Models are auto-downloaded from HuggingFace on first use:

```python
from flashdiffusion import FlashDiffusion

model = FlashDiffusion("runwayml/stable-diffusion-v1-5")
```

## Custom Models

Any HuggingFace diffusers-compatible model can be used:

```python
model = FlashDiffusion("your-username/your-model")
```
