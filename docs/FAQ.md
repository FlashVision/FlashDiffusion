# FAQ

## How much VRAM do I need?

- **SD 1.5**: ~4 GB for inference, ~8 GB for LoRA training
- **SDXL**: ~8 GB for inference, ~16 GB for LoRA training
- **FLUX**: ~24 GB for inference

## Which scheduler should I use?

For most use cases, **DPM++ 2M** offers the best quality-to-speed ratio with 20-25 steps. For instant generation, use **LCM** with 1-4 steps (requires LCM-distilled model).

## Can I use custom models?

Yes, any HuggingFace diffusers-compatible model works:

```python
from flashdiffusion import FlashDiffusion

model = FlashDiffusion("your-username/your-model")
```

## How to use ControlNet?

```python
from flashdiffusion.pipelines import ControlNetPipeline

pipe = ControlNetPipeline(controlnet_type="canny")
images = pipe("a beautiful house", control_image="edges.png")
```

Supported types: canny, depth, pose, scribble, hed, mlsd, seg, normal.

## How to train LoRA?

```bash
flashdiffusion train --model sd15 --data data/my_images --method lora --steps 1000
```

LoRA typically needs 500-2000 steps with 5-20 training images.

## How to improve generation quality?

- Use negative prompts to avoid unwanted artifacts
- Increase `guidance_scale` (7-12) for more prompt-adherent results
- Use more inference steps (30-50) with DDIM or DPM++
- Try different seeds for variation
- Use higher resolution (768x768 or 1024x1024 for SDXL)
