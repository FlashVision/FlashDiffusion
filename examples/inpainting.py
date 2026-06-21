"""
Inpainting Example
====================

Fill masked regions of an image with generated content.
The mask should be white where you want to inpaint and black where to keep.
"""

from flashdiffusion.pipelines import InpaintingPipeline

pipe = InpaintingPipeline(
    model_id="runwayml/stable-diffusion-inpainting",
    device="cuda",
)

images = pipe(
    prompt="a red sports car, photorealistic, high quality",
    image="input.jpg",
    mask="mask.png",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
)

images[0].save("inpainted_output.png")
print(f"Inpainted image saved: inpainted_output.png")
