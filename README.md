<div align="center">

# FlashDiffusion

**Lightweight Diffusion Models — Text-to-Image, Img2Img, Inpainting, ControlNet, and LoRA Fine-Tuning**

[![PyPI](https://img.shields.io/badge/PyPI-flashdiffusion-blue.svg)](https://pypi.org/project/flashdiffusion/)
[![CI](https://github.com/FlashVision/FlashDiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashDiffusion/actions)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ONNX](https://img.shields.io/badge/ONNX-supported-005CED.svg)](https://onnx.ai)
[![LoRA](https://img.shields.io/badge/LoRA-supported-brightgreen.svg)](#lora-fine-tuning)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Documentation](docs/Home.md) · [Quick Start](docs/Quick-Start.md) · [Models](docs/Models.md) · [Schedulers](docs/Schedulers.md) · [LoRA](docs/LoRA-Fine-Tuning.md) · [ControlNet](docs/ControlNet.md)

</div>

---

## What is FlashDiffusion?

FlashDiffusion is a modular, production-ready framework for latent diffusion models. It provides everything you need to **generate**, **fine-tune**, **condition**, and **deploy** diffusion models — from text-to-image generation to inpainting, ControlNet spatial conditioning, and parameter-efficient LoRA fine-tuning.

### Key Features

- **Modular Architecture** — Independent UNet, VAE, text encoder, and scheduler components
- **Multiple Pipelines** — Text-to-image, image-to-image, inpainting, ControlNet generation
- **5 Schedulers** — DDPM, DDIM, Euler/Euler Ancestral, PNDM, DPM-Solver++
- **LoRA Fine-Tuning** — Standard LoRA, QLoRA, DoRA, AdaLoRA, OrthoLoRA, LoRA-FA
- **ControlNet** — Spatial conditioning with edges, depth, pose, segmentation
- **ONNX Export** — Deploy with ONNX Runtime or TensorRT
- **Registry System** — Plug in custom models, schedulers, and pipelines via config
- **Mixed Precision** — AMP training with automatic loss scaling
- **CLI & Python API** — Both command-line and programmatic interfaces

---

## Installation

```bash
pip install flashdiffusion
```

**With extras:**

```bash
pip install flashdiffusion[analytics]    # matplotlib, pandas
pip install flashdiffusion[export]       # onnx, onnxruntime
pip install flashdiffusion[controlnet]   # controlnet-aux
pip install flashdiffusion[all]          # everything
```

**From source:**

```bash
git clone https://github.com/FlashVision/FlashDiffusion.git
cd FlashDiffusion
pip install -e ".[dev,all]"
```

**Verify:**

```bash
flashdiffusion check
```

---

## Usage

### Python API

```python
from flashdiffusion.models.unet.unet2d import UNet2DConditionModel
from flashdiffusion.models.vae.autoencoder import AutoencoderKL
from flashdiffusion.schedulers import DDIMScheduler
import torch

# Build model
unet = UNet2DConditionModel(
    in_channels=4, out_channels=4,
    model_channels=128, channel_mult=(1, 2, 4, 4),
    num_res_blocks=2, context_dim=768,
)
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4)
scheduler = DDIMScheduler()

# Generate
scheduler.set_timesteps(50)
latents = torch.randn(1, 4, 32, 32)
context = torch.randn(1, 77, 768)

unet.eval()
with torch.no_grad():
    for t in scheduler.timesteps:
        noise_pred = unet(latents, t.unsqueeze(0), context=context)
        latents = scheduler.step(noise_pred, t.item(), latents, eta=0.0)

images = vae.decode(latents)
```

### CLI

```bash
# Generate from text
flashdiffusion generate --model model.pth --prompt "a sunset" --steps 50

# Image-to-image
flashdiffusion img2img --model model.pth --image input.png --prompt "oil painting"

# Inpainting
flashdiffusion inpaint --model model.pth --image photo.png --mask mask.png --prompt "blue sky"

# Train
flashdiffusion train --config configs/flashdiffusion_txt2img_256.yaml

# Export to ONNX
flashdiffusion export --model model.pth --output model.onnx

# Benchmark
flashdiffusion benchmark --model model.pth --device cuda
```

---

## Models

| Model | Channels | Multipliers | Parameters | Description |
|-------|----------|-------------|------------|-------------|
| UNet-Tiny | 64 | (1, 2) | ~2M | Testing and CI |
| UNet-Small | 128 | (1, 2, 4) | ~45M | Low-resource training |
| UNet-Base | 128 | (1, 2, 4, 4) | ~120M | Standard (SD 1.x style) |
| UNet-Large | 256 | (1, 2, 4, 4) | ~400M | High-quality generation |
| VAE-KL | 128 | (1, 2, 4, 4) | ~34M | Latent autoencoder |
| ControlNet | 128 | (1, 2, 4, 4) | ~60M | Spatial conditioning |

---

## Schedulers

| Scheduler | Steps | Quality | Speed | Use Case |
|-----------|-------|---------|-------|----------|
| DDPM | 1000 | ★★★★★ | ★☆☆☆☆ | Reference quality |
| DDIM | 20-50 | ★★★★☆ | ★★★★☆ | Fast, reproducible |
| Euler | 20-30 | ★★★★☆ | ★★★★☆ | General purpose |
| Euler Ancestral | 20-30 | ★★★☆☆ | ★★★★☆ | Creative diversity |
| PNDM | 20-50 | ★★★☆☆ | ★★★★☆ | Balanced |
| DPM-Solver++ | 15-25 | ★★★★★ | ★★★★★ | Best quality/speed |

---

## Pipelines

| Pipeline | Input | Output | Description |
|----------|-------|--------|-------------|
| `Txt2ImgPipeline` | Text prompt | Image | Text-conditioned generation |
| `Img2ImgPipeline` | Image + text | Image | Style transfer / transformation |
| `InpaintPipeline` | Image + mask + text | Image | Fill masked regions |
| `ControlNetPipeline` | Hint + text | Image | Spatially-conditioned generation |

---

## Solutions

| Solution | Description |
|----------|-------------|
| `ImageGenerator` | High-level image generation with automatic scheduling |
| `StyleTransfer` | Apply artistic styles to existing images |

---

## Training

### Standard Training

```bash
flashdiffusion train --config configs/flashdiffusion_txt2img_256.yaml
```

### LoRA Fine-Tuning

```bash
flashdiffusion train --config configs/flashdiffusion_lora_finetune.yaml --lora
```

Supported LoRA variants:

| Variant | Description | Trainable Params |
|---------|-------------|-----------------|
| Standard LoRA | Low-rank adaptation | ~1-2% |
| QLoRA | 4-bit quantized base + LoRA | ~1-2% |
| DoRA | Weight-decomposed LoRA | ~1-2% + magnitude |
| AdaLoRA | Adaptive rank allocation | ~1-2% (adaptive) |
| OrthoLoRA | Orthogonality-constrained | ~1-2% |
| LoRA-FA | Frozen-A LoRA | ~0.5-1% |

### ControlNet Training

```bash
flashdiffusion train --config configs/flashdiffusion_controlnet_256.yaml
```

---

## Analytics

```bash
flashdiffusion benchmark --model model.pth --device cuda --resolution 256 --iterations 100
```

Reports:
- Average inference time (ms)
- Throughput (images/sec)
- Peak GPU memory (MB)
- Per-step latency across schedulers

---

## Examples

| Example | Script | Description |
|---------|--------|-------------|
| Text-to-Image | `examples/txt2img.py` | Generate from text prompts |
| Image-to-Image | `examples/img2img.py` | Transform with style guidance |
| Inpainting | `examples/inpainting.py` | Fill masked image regions |
| ControlNet | `examples/controlnet_generate.py` | Spatially-conditioned generation |
| LoRA Training | `examples/train_lora.py` | Parameter-efficient fine-tuning |
| Benchmark | `examples/benchmark_generation.py` | Speed benchmarks |
| ONNX Export | `examples/export_onnx.py` | Export to ONNX format |

```bash
# Quick demo (runs on CPU, no pretrained weights needed)
python examples/txt2img.py --prompt "a mountain landscape" --steps 20
```

---

## Project Structure

```
FlashDiffusion/
├── flashdiffusion/              # Core library
│   ├── models/                  # Model definitions
│   │   ├── unet/                # UNet2DConditionModel, blocks, attention
│   │   └── vae/                 # AutoencoderKL
│   ├── schedulers/              # DDPM, DDIM, Euler, PNDM, DPM-Solver++
│   ├── pipelines/               # Txt2Img, Img2Img, Inpaint, ControlNet
│   ├── engine/                  # Trainer, Validator, Predictor, Exporter
│   ├── nn/                      # Shared blocks (GroupNorm32, timestep_embedding)
│   ├── cfg/                     # Dataclass configuration
│   ├── registry.py              # Pluggable component registry
│   └── cli.py                   # Command-line interface
├── configs/                     # YAML configuration files
│   ├── flashdiffusion_txt2img_256.yaml
│   ├── flashdiffusion_inpaint_256.yaml
│   ├── flashdiffusion_controlnet_256.yaml
│   └── flashdiffusion_lora_finetune.yaml
├── examples/                    # Runnable example scripts
├── tests/                       # Pytest test suite
├── docs/                        # Documentation
├── docker/                      # Dockerfile & docker-compose
├── .github/workflows/ci.yml     # GitHub Actions CI
├── pyproject.toml               # Build configuration
├── CONTRIBUTING.md              # Contribution guide
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Docker

```bash
# Build
docker build -t flashdiffusion -f docker/Dockerfile .

# Run with GPU
docker run --gpus all -it flashdiffusion flashdiffusion check

# Docker Compose
docker compose -f docker/docker-compose.yml up
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FlashVision/FlashDiffusion.git
cd FlashDiffusion
pip install -e ".[dev,all]"
pytest tests/ -v
ruff check flashdiffusion/
```

---

## License

FlashDiffusion is released under the [MIT License](LICENSE).

---

<div align="center">
  <b>Part of the <a href="https://github.com/FlashVision">FlashVision</a> ecosystem</b>
</div>
