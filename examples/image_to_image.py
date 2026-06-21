"""
Image-to-Image Style Transfer
================================

Transform an existing image using a text prompt and strength control.
Lower strength preserves more of the original; higher strength transforms more.
"""

from flashdiffusion.pipelines import Img2ImgPipeline

pipe = Img2ImgPipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
)

images = pipe(
    prompt="watercolor painting, soft colors, artistic",
    image="input.jpg",
    strength=0.7,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
)

images[0].save("styled_output.png")
print(f"Styled image saved: styled_output.png")
