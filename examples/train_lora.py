"""
LoRA Fine-Tuning Example
==========================

Fine-tune Stable Diffusion using LoRA on your own images.
Only ~0.1% of parameters are trained — fast and memory-efficient.

Requirements:
    pip install flashdiffusion
    Prepare a folder of images: data/my_images/
"""

from flashdiffusion import Trainer

trainer = Trainer(
    model_id="runwayml/stable-diffusion-v1-5",
    train_data="data/my_images",
    method="lora",
    max_train_steps=1000,
    learning_rate=1e-4,
    lora_rank=4,
    lora_alpha=4.0,
    batch_size=1,
    gradient_accumulation_steps=4,
    device="cuda",
    save_dir="workspace/lora_model",
    instance_prompt="a photo of sks subject",
)

print("Starting LoRA training...")
metrics = trainer.train()
print(f"Training complete! Final loss: {metrics['final_loss']:.6f}")
print("LoRA weights saved to workspace/lora_model/")
