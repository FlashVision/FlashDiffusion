# Contributing to FlashDiffusion

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashDiffusion.git
cd FlashDiffusion
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashdiffusion/`
4. Run tests: `pytest tests/ -v`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Scheduler

1. Create `flashdiffusion/schedulers/your_scheduler.py`
2. Inherit from `BaseScheduler` and implement `set_timesteps()`, `step()`, `add_noise()`
3. Register with `@SCHEDULERS.register("YourScheduler")`
4. Add to `flashdiffusion/schedulers/__init__.py`

## Adding a New Pipeline

1. Create `flashdiffusion/pipelines/your_pipeline.py`
2. Implement `__call__()` for the generation workflow
3. Register with `@PIPELINES.register("YourPipeline")`
4. Add to `flashdiffusion/pipelines/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add ControlNet depth pipeline`
- `Fix scheduler timestep alignment`
- `Update README with LoRA examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashdiffusion settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
