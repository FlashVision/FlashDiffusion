# Changelog

All notable changes to FlashDiffusion will be documented in this file.

## [1.0.0] — 2026-06-21

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashdiffusion generate`, `train`, `img2img`, `inpaint`, `export`, `benchmark`, `check`, `settings`, `version`
- **Python API** — `FlashDiffusion`, `Trainer`, `Predictor`, `Validator`, `Exporter`
- **Pipelines** — Text-to-Image, Image-to-Image, Inpainting, ControlNet
- **Schedulers** — DDPM, DDIM, DPM++, Euler, LCM
- **LoRA training** — LoRA / DreamBooth / Textual Inversion fine-tuning
- **Models** — Stable Diffusion 1.5, SDXL, FLUX support via HuggingFace diffusers
- **ControlNet** — Canny, Depth, Pose conditioning
- **Solutions** — ImageGenerator, StyleTransfer, Upscaler
- **Analytics** — Benchmark, Profiler, training curve plots
- **Export** — ONNX, TensorRT export support
- **Mixed precision** — AMP (FP16) training with gradient checkpointing
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12)
- **Examples** — 5 runnable example scripts

### Architecture
- HuggingFace diffusers pipeline integration
- UNet2D with cross-attention for noise prediction
- VAE (AutoencoderKL) for latent encoding/decoding
- CLIP text encoder for prompt conditioning
- Pluggable scheduler system with unified API
