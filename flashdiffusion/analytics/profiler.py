"""Profiler — component-wise latency analysis for diffusion pipelines."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union


class Profiler:
    """Profile a diffusion pipeline component by component.

    Measures time spent in text encoding, UNet inference, VAE decoding,
    and scheduling overhead.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
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
        prompt: str = "a beautiful landscape",
        num_inference_steps: int = 30,
    ) -> Dict[str, float]:
        """Profile the pipeline and return per-component timings.

        Returns
        -------
        dict
            ``{"text_encoder_ms", "unet_total_ms", "unet_per_step_ms", "vae_decode_ms", "total_ms"}``
        """
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        vae = pipe.vae
        scheduler = pipe.scheduler

        def sync():
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            encoder_hidden_states = text_encoder(tokens)[0]
        sync()
        text_encoder_ms = (time.perf_counter() - t0) * 1000

        scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(1, unet.config.in_channels, 64, 64, device=self.device, dtype=encoder_hidden_states.dtype)

        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        sync()
        unet_total_ms = (time.perf_counter() - t0) * 1000

        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            vae.decode(latents / 0.18215)
        sync()
        vae_decode_ms = (time.perf_counter() - t0) * 1000

        total_ms = text_encoder_ms + unet_total_ms + vae_decode_ms

        results = {
            "text_encoder_ms": round(text_encoder_ms, 2),
            "unet_total_ms": round(unet_total_ms, 2),
            "unet_per_step_ms": round(unet_total_ms / num_inference_steps, 2),
            "vae_decode_ms": round(vae_decode_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        print(f"{'Component':<20} {'Time (ms)':>10}")
        print("-" * 32)
        for key, val in results.items():
            print(f"  {key:<18} {val:>10.2f}")

        return results

    def summary(self, **kwargs) -> str:
        """Return a human-readable profiling summary."""
        results = self.run(**kwargs)
        lines = [
            f"{'Component':<20} {'Time (ms)':>10}",
            "-" * 32,
        ]
        for key, val in results.items():
            lines.append(f"  {key:<18} {val:>10.2f}")
        return "\n".join(lines)
