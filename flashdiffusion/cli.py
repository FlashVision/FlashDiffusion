"""FlashDiffusion CLI — command-line interface for generation, training, and export."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashDiffusion", "bold") + f" v{_get_version()}")
    print(_colored("Lightweight diffusion model framework", "blue"))
    print()


def _get_version():
    from flashdiffusion import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    try:
        import diffusers
        print(f"  Diffusers:   {diffusers.__version__}")
    except ImportError:
        print("  Diffusers:   Not installed")
    try:
        import transformers
        print(f"  Transformers:{transformers.__version__}")
    except ImportError:
        print("  Transformers:Not installed")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic pipeline."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashdiffusion  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashdiffusion package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashdiffusion package: {e}")
        errors.append(str(e))

    try:
        from flashdiffusion.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashdiffusion.pipelines import Txt2ImgPipeline, Img2ImgPipeline, InpaintingPipeline  # noqa: F401
        print(f"  {_colored('✓', 'green')} pipelines (Txt2Img, Img2Img, Inpainting)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} pipelines: {e}")
        errors.append(str(e))

    try:
        from flashdiffusion.schedulers import DDPMScheduler, DDIMScheduler, EulerScheduler  # noqa: F401
        print(f"  {_colored('✓', 'green')} schedulers (DDPM, DDIM, Euler, ...)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} schedulers: {e}")
        errors.append(str(e))

    try:
        from flashdiffusion.solutions import ImageGenerator, StyleTransfer, Upscaler  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (ImageGenerator, StyleTransfer, Upscaler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashdiffusion.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import diffusers  # noqa: F401
        print(f"  {_colored('✓', 'green')} diffusers ({diffusers.__version__})")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} diffusers: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (generation will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashDiffusion is ready.", "green"))


def cmd_generate(args):
    """Generate images from text prompt."""
    from flashdiffusion.engine.predictor import Predictor
    from flashdiffusion.cfg.config import MODEL_PRESETS

    model_id = MODEL_PRESETS.get(args.model, args.model)

    if args.config:
        from flashdiffusion.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        model_id = cfg.model.model_id

    predictor = Predictor(model_id=model_id, device=args.device)
    images = predictor.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )

    output = args.output or "generated.png"
    images[0].save(output)
    print(f"\n{_colored('✓', 'green')} Image saved: {output}")


def cmd_train(args):
    """Train a LoRA / DreamBooth / Textual Inversion model."""
    from flashdiffusion.engine.trainer import Trainer
    from flashdiffusion.cfg.config import MODEL_PRESETS

    if args.config:
        from flashdiffusion.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        model_id = MODEL_PRESETS.get(args.model, args.model)
        trainer = Trainer(
            model_id=model_id,
            train_data=args.data,
            method=args.method,
            max_train_steps=args.steps,
            lora_rank=args.lora_rank,
            device=args.device,
            save_dir=args.save_dir,
        )

    trainer.train()


def cmd_img2img(args):
    """Run image-to-image generation."""
    from flashdiffusion.pipelines import Img2ImgPipeline
    from flashdiffusion.cfg.config import MODEL_PRESETS

    model_id = MODEL_PRESETS.get(args.model, args.model)
    pipe = Img2ImgPipeline(model_id=model_id, device=args.device)
    images = pipe(
        prompt=args.prompt,
        image=args.image,
        strength=args.strength,
        num_inference_steps=args.steps,
    )

    output = args.output or "img2img_output.png"
    images[0].save(output)
    print(f"\n{_colored('✓', 'green')} Image saved: {output}")


def cmd_inpaint(args):
    """Run inpainting."""
    from flashdiffusion.pipelines import InpaintingPipeline
    from flashdiffusion.cfg.config import MODEL_PRESETS

    model_id = MODEL_PRESETS.get(args.model, args.model)
    pipe = InpaintingPipeline(model_id=model_id, device=args.device)
    images = pipe(
        prompt=args.prompt,
        image=args.image,
        mask=args.mask,
        num_inference_steps=args.steps,
    )

    output = args.output or "inpaint_output.png"
    images[0].save(output)
    print(f"\n{_colored('✓', 'green')} Image saved: {output}")


def cmd_export(args):
    """Export model to ONNX."""
    from flashdiffusion.engine.exporter import Exporter
    from flashdiffusion.cfg.config import MODEL_PRESETS

    model_id = MODEL_PRESETS.get(args.model, args.model)
    exporter = Exporter(model_id=model_id)
    path = exporter.export(output=args.output)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def cmd_benchmark(args):
    """Benchmark model performance."""
    from flashdiffusion.analytics import Benchmark
    from flashdiffusion.cfg.config import MODEL_PRESETS

    model_id = MODEL_PRESETS.get(args.model, args.model)
    bench = Benchmark(model_id=model_id, device=args.device)
    results = bench.run(num_inference_steps=args.steps)

    print(f"\n{'=' * 40}")
    print("  FlashDiffusion Benchmark Results")
    print(f"{'=' * 40}")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 40}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashdiffusion",
        description="FlashDiffusion: Lightweight diffusion model framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashdiffusion check                                 Verify installation
  flashdiffusion generate --prompt "a castle" --model sd15
  flashdiffusion train --model sd15 --data data/ --method lora
  flashdiffusion export --model sd15 --output unet.onnx

Documentation: https://github.com/FlashVision/FlashDiffusion
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate images from text prompt")
    gen_p.add_argument("--prompt", required=True, help="Text prompt for generation")
    gen_p.add_argument("--negative-prompt", default="", help="Negative prompt")
    gen_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID (default: sd15)")
    gen_p.add_argument("--config", default=None, help="Path to YAML config")
    gen_p.add_argument("--steps", type=int, default=30, help="Number of inference steps (default: 30)")
    gen_p.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    gen_p.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_p.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    gen_p.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    gen_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    gen_p.add_argument("--output", default=None, help="Output path (default: generated.png)")

    # train
    train_p = subparsers.add_parser("train", help="Train LoRA / DreamBooth / Textual Inversion")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID")
    train_p.add_argument("--data", default=None, help="Path to training images")
    train_p.add_argument("--method", default="lora", choices=["lora", "dreambooth", "textual_inversion"],
                         help="Training method (default: lora)")
    train_p.add_argument("--steps", type=int, default=1000, help="Max training steps (default: 1000)")
    train_p.add_argument("--lora-rank", type=int, default=4, help="LoRA rank (default: 4)")
    train_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    train_p.add_argument("--save-dir", default="workspace/train", help="Output directory")
    train_p.add_argument("--instance-prompt", default=None, help="Instance prompt for DreamBooth")
    train_p.add_argument("--class-prompt", default=None, help="Class prompt for prior preservation")
    train_p.add_argument("--placeholder-token", default=None, help="Placeholder token for Textual Inversion")

    # img2img
    i2i_p = subparsers.add_parser("img2img", help="Image-to-image generation")
    i2i_p.add_argument("--prompt", required=True, help="Text prompt")
    i2i_p.add_argument("--image", required=True, help="Input image path")
    i2i_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID")
    i2i_p.add_argument("--strength", type=float, default=0.7, help="Strength (default: 0.7)")
    i2i_p.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    i2i_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    i2i_p.add_argument("--output", default=None, help="Output path")

    # inpaint
    inp_p = subparsers.add_parser("inpaint", help="Inpainting")
    inp_p.add_argument("--prompt", required=True, help="Text prompt")
    inp_p.add_argument("--image", required=True, help="Input image path")
    inp_p.add_argument("--mask", required=True, help="Mask image path")
    inp_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID")
    inp_p.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    inp_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    inp_p.add_argument("--output", default=None, help="Output path")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX")
    exp_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID")
    exp_p.add_argument("--output", default="unet.onnx", help="Output path (default: unet.onnx)")

    # benchmark
    bm_p = subparsers.add_parser("benchmark", help="Benchmark model performance")
    bm_p.add_argument("--model", default="sd15", help="Model preset or HuggingFace ID")
    bm_p.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    bm_p.add_argument("--device", default="cuda", help="Device (default: cuda)")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "generate": cmd_generate,
        "train": cmd_train,
        "img2img": cmd_img2img,
        "inpaint": cmd_inpaint,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
