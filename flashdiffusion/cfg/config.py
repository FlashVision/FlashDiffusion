"""
Configuration for FlashDiffusion.

Default settings target Stable Diffusion v1.5.  Override via YAML configs
or keyword arguments to ``get_config()``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset paths for LoRA / DreamBooth training."""

    train_data: str = "data/train"
    instance_prompt: str = "a photo of sks subject"
    class_prompt: Optional[str] = None
    class_data_dir: Optional[str] = None
    resolution: int = 512
    center_crop: bool = True
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "FlashDiffusion"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None
    resolution: Tuple[int, int] = (512, 512)

    scheduler: str = "ddim"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

    use_safetensors: bool = True
    torch_dtype: str = "float16"


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    method: str = "lora"
    max_train_steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100

    save_dir: str = "workspace/default_experiment"
    save_every_n_steps: int = 500
    resume: Optional[str] = None

    enable_gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    use_ema: bool = False
    ema_decay: float = 0.9999

    lora_rank: int = 4
    lora_alpha: float = 4.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"])

    prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100

    placeholder_token: str = "<my-concept>"
    initializer_token: str = "object"
    num_vectors: int = 1

    seed: int = 42


@dataclass
class Config:
    """Top-level configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


MODEL_PRESETS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-dev",
}


def get_config(
    model_id: Optional[str] = None,
    resolution: int = 512,
    **overrides,
) -> Config:
    """Return configuration for a given model.

    Args:
        model_id: HuggingFace model ID or preset name (sd15, sdxl, flux).
        resolution: Image resolution.
        **overrides: Additional overrides applied to the Config.
    """
    cfg = Config()

    if model_id is not None:
        resolved = MODEL_PRESETS.get(model_id, model_id)
        cfg.model.model_id = resolved

    cfg.model.resolution = (resolution, resolution)
    cfg.data.resolution = resolution

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    YAML structure mirrors the Config dataclass hierarchy:
        model:
          model_id: "runwayml/stable-diffusion-v1-5"
          resolution: [512, 512]
        data:
          train_data: data/train
        train:
          method: lora
          max_train_steps: 1000
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "model" in raw:
        for key, value in raw["model"].items():
            if key == "resolution" and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)

    if "data" in raw:
        for key, value in raw["data"].items():
            if hasattr(cfg.data, key):
                setattr(cfg.data, key, value)

    if "train" in raw:
        for key, value in raw["train"].items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)

    return cfg
