"""FlashDiffusion Validator — compute FID and CLIP score metrics."""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class Validator:
    """Evaluate generated image quality using FID and CLIP score.

    Example::

        from flashdiffusion import Validator

        val = Validator(model_id="runwayml/stable-diffusion-v1-5")
        results = val.validate(
            prompts=["a cat", "a dog"],
            reference_dir="data/reference",
        )
        print(f"CLIP Score: {results['clip_score']:.4f}")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        num_inference_steps: int = 30,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def validate(
        self,
        prompts: List[str],
        reference_dir: Optional[str] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run validation and compute quality metrics.

        Args:
            prompts: List of text prompts to generate images from.
            reference_dir: Directory of reference images for FID computation.
            num_images_per_prompt: Images to generate per prompt.
            guidance_scale: Guidance scale for generation.
            seed: Random seed.

        Returns:
            Dict with keys: clip_score, num_images (and optionally fid).
        """
        from flashdiffusion.engine.predictor import Predictor
        from flashdiffusion.utils.metrics import compute_clip_score

        predictor = Predictor(model_id=self.model_id, device=str(self.device))

        all_images = []
        for prompt in prompts:
            images = predictor.generate(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_images=num_images_per_prompt,
            )
            all_images.extend(images)

        clip_score = compute_clip_score(all_images, prompts)

        result = {
            "clip_score": clip_score,
            "num_images": len(all_images),
        }

        if reference_dir is not None:
            try:
                from flashdiffusion.utils.metrics import compute_fid

                fid = compute_fid(all_images, reference_dir)
                result["fid"] = fid
            except ImportError:
                logger.warning("clean-fid not installed, skipping FID computation")

        logger.info("Validation: CLIP Score = %.4f, Images = %d", result["clip_score"], result["num_images"])
        return result
