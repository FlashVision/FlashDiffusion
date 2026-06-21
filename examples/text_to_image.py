"""
Generate Images from Text Prompts
===================================

Load a diffusion model and generate images from text descriptions.
Results are saved as PNG files.
"""

from flashdiffusion import Predictor

predictor = Predictor(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
)

images = predictor.generate(
    prompt="a castle on a hill, sunset, oil painting, detailed",
    negative_prompt="blurry, low quality, watermark",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
)

images[0].save("castle.png")
print(f"Image saved: castle.png ({images[0].size[0]}x{images[0].size[1]})")
