"""Quality metrics for generated images — FID, CLIP Score, Inception Score."""

from typing import List, Optional

import numpy as np
from PIL import Image


def compute_clip_score(
    images: List[Image.Image],
    prompts: List[str],
) -> float:
    """Compute CLIP score between generated images and text prompts.

    Measures how well generated images match their text descriptions.
    Higher is better (typically 20-35 range for good generations).

    Args:
        images: List of generated PIL Images.
        prompts: List of text prompts used for generation.

    Returns:
        Average CLIP score across all image-prompt pairs.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        scores = []
        for image, prompt in zip(images, prompts * (len(images) // len(prompts) + 1)):
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits_per_image.item()
            scores.append(logits)

        return float(np.mean(scores))

    except ImportError:
        return 0.0


def compute_fid(
    generated_images: List[Image.Image],
    reference_dir: str,
) -> float:
    """Compute Frechet Inception Distance between generated and reference images.

    Lower FID is better (0 = identical distributions).

    Args:
        generated_images: List of generated PIL Images.
        reference_dir: Path to directory of reference images.

    Returns:
        FID score.
    """
    try:
        import tempfile
        import os
        from cleanfid import fid

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, img in enumerate(generated_images):
                img.save(os.path.join(tmp_dir, f"gen_{i:04d}.png"))

            score = fid.compute_fid(tmp_dir, reference_dir)
            return float(score)

    except ImportError:
        raise ImportError(
            "clean-fid is required for FID computation. "
            "Install with: pip install clean-fid"
        )


def compute_inception_score(
    images: List[Image.Image],
    splits: int = 10,
) -> float:
    """Compute Inception Score for generated images.

    Higher IS indicates better quality and diversity.

    Args:
        images: List of generated PIL Images.
        splits: Number of splits for IS computation.

    Returns:
        Inception Score (mean).
    """
    try:
        import torch
        from torchvision.models import inception_v3
        import torchvision.transforms as T

        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()

        transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        preds = []
        for img in images:
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                pred = torch.softmax(model(tensor), dim=1)
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        split_scores = []
        n = len(preds)
        for k in range(splits):
            part = preds[k * n // splits: (k + 1) * n // splits]
            py = np.mean(part, axis=0, keepdims=True)
            scores = np.sum(part * np.log(part / py + 1e-10), axis=1)
            split_scores.append(np.exp(np.mean(scores)))

        return float(np.mean(split_scores))

    except ImportError:
        return 0.0
