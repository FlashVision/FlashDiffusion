# LoRA Fine-Tuning

## Overview

LoRA (Low-Rank Adaptation) freezes the base diffusion model and trains only small low-rank adapters in the UNet attention layers. This dramatically reduces training time and memory while maintaining quality.

## Usage

```python
from flashdiffusion import Trainer

trainer = Trainer(
    model_id="runwayml/stable-diffusion-v1-5",
    train_data="data/my_images",
    method="lora",
    lora_rank=4,
    lora_alpha=4.0,
    max_train_steps=1000,
)
trainer.train()
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lora_rank` | Rank of LoRA matrices (lower = fewer params) | 4 |
| `lora_alpha` | Scaling factor (alpha/rank = effective scale) | 4.0 |
| `lora_target_modules` | UNet modules to apply LoRA to | to_q, to_v, to_k, to_out.0 |

## Loading LoRA Weights

```python
from flashdiffusion import Predictor

predictor = Predictor(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_weights="workspace/train/lora_final.safetensors",
)
images = predictor.generate("a photo of sks subject")
```

## DreamBooth

Subject-specific fine-tuning with prior preservation:

```bash
flashdiffusion train --model sd15 --data data/my_subject --method dreambooth \
  --instance-prompt "a photo of sks dog" --class-prompt "a photo of dog"
```

## Textual Inversion

Learn new concepts as text embeddings:

```bash
flashdiffusion train --model sd15 --data data/concept --method textual_inversion \
  --placeholder-token "<my-style>"
```
