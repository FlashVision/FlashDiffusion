"""FlashDiffusion Trainer — LoRA, DreamBooth, and Textual Inversion training."""

import os
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdiffusion.models.lora import apply_lora, get_lora_state_dict
from flashdiffusion.utils import setup_logger, AverageMeter
from flashdiffusion.utils.checkpoint import save_checkpoint
from flashdiffusion.engine.callbacks import CallbackList, Callback

logger = logging.getLogger(__name__)


class Trainer:
    """High-level trainer for diffusion model fine-tuning.

    Supports LoRA, DreamBooth, and Textual Inversion training methods.

    Example::

        from flashdiffusion import Trainer

        trainer = Trainer(
            model_id="runwayml/stable-diffusion-v1-5",
            train_data="data/my_images",
            method="lora",
            max_train_steps=1000,
        )
        trainer.train()
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        train_data: Optional[str] = None,
        method: str = "lora",
        max_train_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        resolution: int = 512,
        device: str = "cuda",
        save_dir: str = "workspace/train",
        save_every_n_steps: int = 500,
        mixed_precision: str = "fp16",
        gradient_checkpointing: bool = True,
        seed: int = 42,
        # LoRA
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_target_modules: Optional[List[str]] = None,
        # DreamBooth
        instance_prompt: str = "a photo of sks subject",
        class_prompt: Optional[str] = None,
        class_data_dir: Optional[str] = None,
        prior_preservation: bool = False,
        prior_loss_weight: float = 1.0,
        # Textual Inversion
        placeholder_token: str = "<my-concept>",
        initializer_token: str = "object",
        # EMA
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        # Config override
        config: Any = None,
    ):
        self.model_id = model_id
        self.train_data = train_data
        self.method = method
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.resolution = resolution
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.seed = seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["to_q", "to_v", "to_k", "to_out.0"]
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.class_data_dir = class_data_dir
        self.prior_preservation = prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if config is not None:
            self._apply_config(config)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = setup_logger("FlashDiffusion", self.save_dir)
        self.callbacks = CallbackList()

    def _apply_config(self, cfg):
        """Apply Config dataclass overrides."""
        if hasattr(cfg, "model"):
            self.model_id = cfg.model.model_id
            self.resolution = (
                cfg.model.resolution[0] if isinstance(cfg.model.resolution, tuple) else cfg.model.resolution
            )
        if hasattr(cfg, "train"):
            self.method = cfg.train.method
            self.max_train_steps = cfg.train.max_train_steps
            self.learning_rate = cfg.train.learning_rate
            self.batch_size = cfg.train.batch_size
            self.gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
            self.save_dir = cfg.train.save_dir
            self.lora_rank = cfg.train.lora_rank
            self.lora_alpha = cfg.train.lora_alpha
            self.seed = cfg.train.seed
        if hasattr(cfg, "data"):
            self.train_data = cfg.data.train_data
            self.instance_prompt = cfg.data.instance_prompt

    def add_callback(self, callback: Callback) -> None:
        """Register a training callback."""
        self.callbacks.add(callback)

    def train(self) -> Dict[str, float]:
        """Run the training loop. Returns dict with final metrics."""
        self._logger.info("=" * 60)
        self._logger.info("FlashDiffusion Training")
        self._logger.info("=" * 60)
        self._logger.info(f"Model: {self.model_id}")
        self._logger.info(f"Method: {self.method}")
        self._logger.info(f"Device: {self.device}")
        self._logger.info(f"Steps: {self.max_train_steps}, LR: {self.learning_rate}")
        self._logger.info(f"Resolution: {self.resolution}, Batch: {self.batch_size}")

        if self.method == "lora":
            return self._train_lora()
        elif self.method == "dreambooth":
            return self._train_dreambooth()
        elif self.method == "textual_inversion":
            return self._train_textual_inversion()
        else:
            raise ValueError(f"Unknown training method: {self.method}")

    def _train_lora(self) -> Dict[str, float]:
        """LoRA fine-tuning — train low-rank adapters on UNet attention layers."""
        from diffusers import StableDiffusionPipeline, DDPMScheduler

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        unet = pipe.unet.to(self.device)
        vae = pipe.vae.to(self.device)
        text_encoder = pipe.text_encoder.to(self.device)
        tokenizer = pipe.tokenizer
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        unet = apply_lora(unet, rank=self.lora_rank, alpha=self.lora_alpha, target_modules=self.lora_target_modules)

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=1e-2)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_train_steps)

        from flashdiffusion.data import create_dataloader

        dataloader = create_dataloader(
            data_dir=self.train_data,
            prompt=self.instance_prompt,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            resolution=self.resolution,
        )

        use_amp = self.mixed_precision == "fp16" and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        self._logger.info("Starting LoRA training...")
        self.callbacks.fire("on_train_start", self)

        loss_meter = AverageMeter("Loss")
        global_step = 0
        unet.train()

        while global_step < self.max_train_steps:
            for batch in dataloader:
                if global_step >= self.max_train_steps:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise) / self.gradient_accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                loss_meter.update(loss.item() * self.gradient_accumulation_steps)
                global_step += 1

                if global_step % 50 == 0:
                    self._logger.info(f"  Step {global_step}/{self.max_train_steps} | Loss: {loss_meter.avg:.6f}")

                if global_step % self.save_every_n_steps == 0:
                    self._save_lora(unet, global_step)

        self._save_lora(unet, global_step, final=True)

        final_metrics = {"final_loss": loss_meter.avg, "total_steps": global_step}
        self.callbacks.fire("on_train_end", self, final_metrics)

        self._logger.info("=" * 60)
        self._logger.info("Training Complete!")
        self._logger.info(f"Final Loss: {loss_meter.avg:.6f} | Steps: {global_step}")
        self._logger.info("=" * 60)

        return final_metrics

    def _train_dreambooth(self) -> Dict[str, float]:
        """DreamBooth training — subject-specific fine-tuning."""
        self._logger.info("DreamBooth training — fine-tuning UNet on subject images")

        from diffusers import StableDiffusionPipeline, DDPMScheduler

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        unet = pipe.unet.to(self.device)
        vae = pipe.vae.to(self.device)
        text_encoder = pipe.text_encoder.to(self.device)
        tokenizer = pipe.tokenizer
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        optimizer = torch.optim.AdamW(unet.parameters(), lr=self.learning_rate, weight_decay=1e-2)

        from flashdiffusion.data import create_dataloader

        dataloader = create_dataloader(
            data_dir=self.train_data,
            prompt=self.instance_prompt,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            resolution=self.resolution,
            dataset_type="dreambooth",
            instance_prompt=self.instance_prompt,
            class_data_dir=self.class_data_dir,
            class_prompt=self.class_prompt,
        )

        loss_meter = AverageMeter("Loss")
        global_step = 0
        unet.train()

        while global_step < self.max_train_steps:
            for batch in dataloader:
                if global_step >= self.max_train_steps:
                    break

                pixel_values = batch["instance_images"].to(self.device)
                input_ids = batch["instance_input_ids"].to(self.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())
                global_step += 1

                if global_step % 50 == 0:
                    self._logger.info(f"  Step {global_step}/{self.max_train_steps} | Loss: {loss_meter.avg:.6f}")

        save_checkpoint(
            unet, optimizer, global_step, loss_meter.avg, os.path.join(self.save_dir, "dreambooth_unet.pth")
        )
        return {"final_loss": loss_meter.avg, "total_steps": global_step}

    def _train_textual_inversion(self) -> Dict[str, float]:
        """Textual Inversion — learn new concept embeddings."""
        self._logger.info("Textual Inversion — learning new concept embedding")

        from diffusers import StableDiffusionPipeline, DDPMScheduler

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(self.device)
        unet = pipe.unet.to(self.device)
        vae = pipe.vae.to(self.device)
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        tokenizer.add_tokens(self.placeholder_token)
        text_encoder.resize_token_embeddings(len(tokenizer))

        placeholder_token_id = tokenizer.convert_tokens_to_ids(self.placeholder_token)
        initializer_token_id = tokenizer.encode(self.initializer_token, add_special_tokens=False)[0]

        text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = (
            text_encoder.get_input_embeddings().weight.data[initializer_token_id].clone()
        )

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder.get_input_embeddings().weight.requires_grad = True

        optimizer = torch.optim.AdamW([text_encoder.get_input_embeddings().weight], lr=self.learning_rate)

        from flashdiffusion.data import create_dataloader

        dataloader = create_dataloader(
            data_dir=self.train_data,
            prompt=f"a photo of {self.placeholder_token}",
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            resolution=self.resolution,
        )

        loss_meter = AverageMeter("Loss")
        global_step = 0

        while global_step < self.max_train_steps:
            for batch in dataloader:
                if global_step >= self.max_train_steps:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]

                with torch.no_grad():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    mask = torch.ones(len(tokenizer), dtype=torch.bool)
                    mask[placeholder_token_id] = False
                    text_encoder.get_input_embeddings().weight.data[mask] = (
                        text_encoder.get_input_embeddings().weight.data[mask].detach()
                    )

                loss_meter.update(loss.item())
                global_step += 1

                if global_step % 50 == 0:
                    self._logger.info(f"  Step {global_step}/{self.max_train_steps} | Loss: {loss_meter.avg:.6f}")

        embed_path = os.path.join(self.save_dir, "learned_embeds.bin")
        learned = text_encoder.get_input_embeddings().weight.data[placeholder_token_id]
        torch.save({self.placeholder_token: learned}, embed_path)
        self._logger.info(f"Saved learned embedding: {embed_path}")

        return {"final_loss": loss_meter.avg, "total_steps": global_step}

    def _save_lora(self, unet, step, final=False):
        prefix = "final" if final else f"step_{step}"
        lora_path = os.path.join(self.save_dir, f"lora_{prefix}.safetensors")
        state = get_lora_state_dict(unet)
        torch.save(state, lora_path)
        self._logger.info(f"LoRA weights saved: {lora_path}")
