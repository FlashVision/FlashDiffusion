# Quick Start

## Generate an image

```python
from flashdiffusion import Predictor

predictor = Predictor(model_id="runwayml/stable-diffusion-v1-5", device="cuda")
images = predictor.generate(
    prompt="a castle on a hill, sunset, oil painting",
    seed=42,
)
images[0].save("castle.png")
```

## Train LoRA

```python
from flashdiffusion import Trainer

trainer = Trainer(
    model_id="runwayml/stable-diffusion-v1-5",
    train_data="data/my_images",
    method="lora",
    max_train_steps=1000,
)
trainer.train()
```

## Export to ONNX

```python
from flashdiffusion import Exporter

exporter = Exporter(model_id="runwayml/stable-diffusion-v1-5")
exporter.export(output="unet.onnx")
```

## CLI

```bash
flashdiffusion generate --prompt "a beautiful landscape" --model sd15 --steps 30
flashdiffusion train --model sd15 --data data/my_images --method lora --steps 1000
flashdiffusion export --model sd15 --output unet.onnx
```
