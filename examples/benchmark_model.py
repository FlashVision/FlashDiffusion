"""
Benchmark Model Performance
=============================

Measure generation speed, latency, and VRAM usage.
Useful for comparing models and schedulers before deployment.
"""

from flashdiffusion.analytics import Benchmark

bench = Benchmark(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
)

results = bench.run(
    prompt="a beautiful landscape, oil painting",
    num_inference_steps=30,
    warmup=1,
    iterations=3,
)

print("=" * 45)
print("  FlashDiffusion Benchmark Results")
print("=" * 45)
print(f"  Model:          {results['model']}")
print(f"  Resolution:     {results['resolution']}")
print(f"  Steps:          {results['steps']}")
print(f"  Images/sec:     {results['images_per_sec']:.3f}")
print(f"  Latency:        {results['latency_ms']:.1f} ms")
print(f"  VRAM:           {results['vram_mb']:.1f} MB")
print("=" * 45)
