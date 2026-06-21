"""Benchmark — measure FlashDiffusion generation speed and resource usage."""

from __future__ import annotations

import time
from typing import Any, Dict, List


class Benchmark:
    """Benchmark diffusion model generation speed and resource usage.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or preset name.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device

    def run(
        self,
        prompt: str = "a beautiful landscape, oil painting",
        num_inference_steps: int = 30,
        warmup: int = 1,
        iterations: int = 3,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        """Run a generation speed benchmark.

        Returns
        -------
        dict
            ``{"images_per_sec": …, "latency_ms": …, "vram_mb": …, "steps": …}``
        """
        import torch
        from flashdiffusion.engine.predictor import Predictor

        predictor = Predictor(model_id=self.model_id, device=self.device)

        for _ in range(warmup):
            predictor.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                seed=42,
            )

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            predictor.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                seed=42,
            )

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / iterations) * 1000
        images_per_sec = iterations / elapsed

        vram_mb = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            "images_per_sec": round(images_per_sec, 3),
            "latency_ms": round(latency_ms, 1),
            "vram_mb": round(vram_mb, 1),
            "steps": num_inference_steps,
            "resolution": f"{width}x{height}",
            "model": self.model_id,
        }

    def compare(
        self,
        model_ids: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Compare multiple models side by side.

        Parameters
        ----------
        model_ids : list[str]
            Additional model IDs to compare against the primary model.

        Returns
        -------
        list[dict]
            One result dict per model (primary model first).
        """
        all_ids = [self.model_id] + model_ids
        results = []
        for mid in all_ids:
            bm = Benchmark(mid, device=self.device)
            res = bm.run(**kwargs)
            results.append(res)
        return results
