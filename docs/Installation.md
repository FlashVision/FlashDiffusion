# Installation

## From PyPI

```bash
pip install flashdiffusion
```

## With extras

```bash
pip install "flashdiffusion[all]"          # Everything
pip install "flashdiffusion[analytics]"    # Benchmarking, plots
pip install "flashdiffusion[export]"       # ONNX export
pip install "flashdiffusion[quality]"      # FID, CLIP score
pip install "flashdiffusion[controlnet]"   # ControlNet preprocessors
```

## From source

```bash
git clone https://github.com/FlashVision/FlashDiffusion.git
cd FlashDiffusion
pip install -e ".[all]"
```

## Verify

```bash
flashdiffusion check
flashdiffusion version
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- HuggingFace diffusers >= 0.25
- HuggingFace transformers >= 4.30
