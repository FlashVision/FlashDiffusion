"""Analytics — benchmarking, profiling and visualization tools for FlashDiffusion."""

from flashdiffusion.analytics.benchmark import Benchmark
from flashdiffusion.analytics.profiler import Profiler
from flashdiffusion.analytics.plots import plot_training_curves, plot_generation_grid

__all__ = [
    "Benchmark",
    "Profiler",
    "plot_training_curves",
    "plot_generation_grid",
]
