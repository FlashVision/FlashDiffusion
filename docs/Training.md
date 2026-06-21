# Training

## LoRA Fine-Tuning

```bash
flashdiffusion train --model sd15 --data data/my_images --method lora --steps 1000
```

## DreamBooth

```bash
flashdiffusion train --model sd15 --data data/my_subject --method dreambooth \
  --instance-prompt "a photo of sks dog" --class-prompt "a photo of dog"
```

## Textual Inversion

```bash
flashdiffusion train --model sd15 --data data/concept --method textual_inversion \
  --placeholder-token "<my-style>"
```

## Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model preset or HuggingFace ID | sd15 |
| `--data` | Path to training images | required |
| `--method` | Training method (lora, dreambooth, textual_inversion) | lora |
| `--steps` | Max training steps | 1000 |
| `--lora-rank` | LoRA rank | 4 |
| `--device` | Device (cuda/cpu) | cuda |
| `--save-dir` | Output directory | workspace/train |

## Config-driven Training

```bash
flashdiffusion train --config configs/flashdiffusion_lora_training.yaml
```

## Features

- AMP (FP16) mixed precision training
- Gradient accumulation and checkpointing
- EMA weight averaging
- Cosine learning rate schedule with warmup
- Periodic checkpoint saving
