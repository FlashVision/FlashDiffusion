"""Comprehensive tests for FlashDiffusion — pipelines, schedulers, models, adapters, video, upscaling, CLI."""

from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Scheduler tests (extended beyond existing)
# ---------------------------------------------------------------------------
class TestSchedulerAlphas:
    def test_ddpm_alphas_cumprod(self):
        from flashdiffusion.schedulers import DDPMScheduler

        s = DDPMScheduler(num_train_timesteps=100)
        assert hasattr(s, "alphas_cumprod")
        assert s.alphas_cumprod[0] > s.alphas_cumprod[-1]

    def test_ddim_alphas(self):
        from flashdiffusion.schedulers import DDIMScheduler

        s = DDIMScheduler(num_train_timesteps=100)
        assert (s.alphas_cumprod >= 0).all()
        assert (s.alphas_cumprod <= 1).all()

    def test_euler_sigmas(self):
        from flashdiffusion.schedulers import EulerScheduler

        s = EulerScheduler(num_train_timesteps=100)
        s.set_timesteps(20)
        assert len(s.timesteps) == 20

    def test_euler_ancestral_step(self):
        from flashdiffusion.schedulers import EulerAncestralScheduler

        s = EulerAncestralScheduler(num_train_timesteps=100)
        s.set_timesteps(10)
        sample = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(sample)
        t = s.timesteps[0].item()
        out = s.step(noise, t, sample)
        assert out.shape == sample.shape

    def test_lcm_fewer_steps(self):
        from flashdiffusion.schedulers import LCMScheduler

        s = LCMScheduler(num_train_timesteps=1000)
        s.set_timesteps(4)
        assert len(s.timesteps) == 4

    def test_dpm_plus_step(self):
        from flashdiffusion.schedulers import DPMPlusPlusScheduler

        s = DPMPlusPlusScheduler(num_train_timesteps=1000)
        s.set_timesteps(15)
        sample = torch.randn(1, 4, 16, 16)
        noise = torch.randn_like(sample)
        t = s.timesteps[0].item()
        out = s.step(noise, t, sample)
        assert out.shape == sample.shape

    def test_ddpm_step(self):
        from flashdiffusion.schedulers import DDPMScheduler

        s = DDPMScheduler(num_train_timesteps=1000)
        s.set_timesteps(20)
        sample = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(sample)
        t = s.timesteps[0].item()
        out = s.step(noise, t, sample)
        assert out.shape == sample.shape


# ---------------------------------------------------------------------------
# SD3 / FLUX (MMDiT, flow matching) — extended
# ---------------------------------------------------------------------------
class TestSD3Extended:
    def test_mmdit_different_seq_lens(self):
        from flashdiffusion.models.sd3 import MMDiTBlock

        block = MMDiTBlock(hidden_size=32, num_heads=4)
        img = torch.randn(1, 8, 32)
        txt = torch.randn(1, 4, 32)
        t = torch.randn(1, 32)
        img_out, txt_out = block(img, txt, t)
        assert img_out.shape == (1, 8, 32)
        assert txt_out.shape == (1, 4, 32)

    def test_flow_matching_noise_interpolation(self):
        from flashdiffusion.models.sd3 import FlowMatchingScheduler

        s = FlowMatchingScheduler()
        original = torch.ones(1, 4, 4, 4)
        noise = torch.zeros(1, 4, 4, 4)
        noisy = s.add_noise(original, noise, sigma=0.5)
        expected = 0.5 * original + 0.5 * noise
        assert torch.allclose(noisy, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# SDXL
# ---------------------------------------------------------------------------
class TestSDXLExtended:
    def test_time_ids_various_sizes(self):
        from flashdiffusion.models.sdxl.sdxl_unet import SDXLUNetWrapper

        for size in [(512, 512), (768, 768), (1024, 1024)]:
            time_ids = SDXLUNetWrapper.compute_time_ids(
                original_size=size,
                crop_coords=(0, 0),
                target_size=size,
                device="cpu",
            )
            assert time_ids.shape == (1, 6)
            assert time_ids[0, 0].item() == size[0]

    def test_dual_text_encoder_init(self):
        from flashdiffusion.models.sdxl.dual_text_encoder import SDXLDualTextEncoder

        dte = SDXLDualTextEncoder(model_id="fake/model")
        assert dte._encoder_1 is None
        assert dte._encoder_2 is None


# ---------------------------------------------------------------------------
# Adapters: IP-Adapter, InstantID
# ---------------------------------------------------------------------------
class TestAdapters:
    def test_decoupled_cross_attention_shapes(self):
        from flashdiffusion.adapters.ip_adapter import DecoupledCrossAttention

        for h, c, n in [(64, 32, 4), (128, 64, 8)]:
            attn = DecoupledCrossAttention(hidden_size=h, cross_attention_dim=c, num_heads=n)
            query = torch.randn(1, 16, h)
            img = torch.randn(1, 4, c)
            out = attn(query, img)
            assert out.shape == (1, 16, h)

    def test_identity_net_variable_tokens(self):
        from flashdiffusion.adapters.instant_id import IdentityNet

        for num_tokens in [4, 16, 32]:
            net = IdentityNet(face_embed_dim=128, cross_attention_dim=256, num_tokens=num_tokens)
            emb = torch.randn(2, 128)
            tokens = net(emb)
            assert tokens.shape == (2, num_tokens, 256)


# ---------------------------------------------------------------------------
# LoRA for diffusion
# ---------------------------------------------------------------------------
class TestDiffusionLoRA:
    def test_lora_linear_trainable_params(self):
        from flashdiffusion.models.lora import LoRALinear

        layer = LoRALinear(64, 64, rank=4, alpha=4.0)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in layer.parameters())
        assert trainable < total

    def test_lora_different_ranks(self):
        from flashdiffusion.models.lora import LoRALinear

        for rank in [2, 4, 8, 16]:
            layer = LoRALinear(64, 64, rank=rank)
            x = torch.randn(1, 64)
            out = layer(x)
            assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# Video: AnimateDiff, SVD
# ---------------------------------------------------------------------------
class TestVideoExtended:
    def test_temporal_attention_gradient(self):
        from flashdiffusion.video.animatediff import TemporalAttention

        attn = TemporalAttention(channels=32, num_heads=4, num_frames=4)
        x = torch.randn(8, 32, 4, 4, requires_grad=True)
        out = attn(x, num_frames=4)
        out.sum().backward()
        assert x.grad is not None

    def test_animatediff_module_keys(self):
        from flashdiffusion.video.animatediff import AnimateDiffMotionModule

        module = AnimateDiffMotionModule(channels_list=[32, 64], num_frames=4, num_heads=4)
        assert "down_0" in module.temporal_attentions
        assert "down_1" in module.temporal_attentions
        assert "mid" in module.temporal_attentions

    def test_svd_pipeline_init(self):
        from flashdiffusion.video.svd import SVDPipeline

        pipe = SVDPipeline(model_id="fake/model", device="cpu")
        assert pipe._pipe is None
        assert pipe.torch_dtype == torch.float32

    def test_temporal_unet_wrapper_init(self):
        from flashdiffusion.video.svd import TemporalUNetWrapper

        wrapper = TemporalUNetWrapper(model_id="fake/model")
        assert wrapper._unet is None


# ---------------------------------------------------------------------------
# Upscaling: ESRGAN
# ---------------------------------------------------------------------------
class TestESRGANExtended:
    def test_rrdbnet_gradient_flow(self):
        from flashdiffusion.upscaling.esrgan import RRDBNet

        model = RRDBNet(in_channels=3, out_channels=3, num_features=16, num_blocks=1, growth_rate=8, scale=2)
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_dense_block_residual(self):
        from flashdiffusion.upscaling.esrgan import DenseBlock

        block = DenseBlock(channels=16, growth_rate=8)
        x = torch.randn(1, 16, 4, 4)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# ControlNet
# ---------------------------------------------------------------------------
class TestControlNetWrapper:
    def test_model_mapping(self):
        from flashdiffusion.models.controlnet.controlnet import CONTROLNET_MODELS

        assert "canny" in CONTROLNET_MODELS
        assert "depth" in CONTROLNET_MODELS
        assert "pose" in CONTROLNET_MODELS

    def test_sdxl_model_mapping(self):
        from flashdiffusion.models.controlnet.controlnet import SDXL_CONTROLNET_MODELS

        assert "canny" in SDXL_CONTROLNET_MODELS

    def test_controlnet_wrapper_init(self):
        from flashdiffusion.models.controlnet.controlnet import ControlNetWrapper

        wrapper = ControlNetWrapper(controlnet_type="canny")
        assert wrapper.controlnet_type == "canny"
        assert wrapper._model is None

    def test_sdxl_controlnet_wrapper_init(self):
        from flashdiffusion.models.controlnet.controlnet import SDXLControlNetWrapper

        wrapper = SDXLControlNetWrapper(controlnet_type="depth")
        assert wrapper.controlnet_type == "depth"


# ---------------------------------------------------------------------------
# Pipelines (init only, no HF model loading)
# ---------------------------------------------------------------------------
class TestPipelineInit:
    def test_txt2img_init(self):
        from flashdiffusion.pipelines.txt2img import Txt2ImgPipeline

        pipe = Txt2ImgPipeline(model_id="fake/model", device="cpu")
        assert pipe._pipe is None
        assert pipe.model_id == "fake/model"

    def test_img2img_init(self):
        from flashdiffusion.pipelines.img2img import Img2ImgPipeline

        pipe = Img2ImgPipeline(model_id="fake/model", device="cpu")
        assert pipe._pipe is None

    def test_inpainting_init(self):
        from flashdiffusion.pipelines.inpainting import InpaintingPipeline

        pipe = InpaintingPipeline(model_id="fake/model", device="cpu")
        assert pipe._pipe is None

    def test_instruct_pix2pix_init(self):
        from flashdiffusion.pipelines.instruct_pix2pix import InstructPix2PixPipeline

        pipe = InstructPix2PixPipeline(model_id="fake/model", device="cpu")
        assert pipe._pipe is None


# ---------------------------------------------------------------------------
# UNet, VAE, CLIP wrappers (init only)
# ---------------------------------------------------------------------------
class TestModelWrappers:
    def test_unet_wrapper_init(self):
        from flashdiffusion.models.unet.unet_2d import UNet2DConditionWrapper

        wrapper = UNet2DConditionWrapper(model_id="fake/model")
        assert wrapper._unet is None
        assert wrapper.model_id == "fake/model"

    def test_vae_wrapper_init(self):
        from flashdiffusion.models.vae.autoencoder import AutoencoderWrapper

        wrapper = AutoencoderWrapper(model_id="fake/model")
        assert wrapper._vae is None
        assert wrapper.LATENT_SCALE_FACTOR == pytest.approx(0.18215)

    def test_clip_wrapper_init(self):
        from flashdiffusion.models.text_encoder.clip_encoder import CLIPTextEncoderWrapper

        wrapper = CLIPTextEncoderWrapper(model_id="fake/model")
        assert wrapper._encoder is None
        assert wrapper._tokenizer is None


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class TestDiffusionLosses:
    def test_diffusion_loss_import(self):
        from flashdiffusion.losses import diffusion_loss  # noqa: F401

    def test_perceptual_loss_import(self):
        from flashdiffusion.losses import perceptual_loss  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestDiffusionCLI:
    def test_version_command(self):
        from flashdiffusion.cli import cmd_version

        cmd_version(SimpleNamespace())

    def test_colored_helper(self):
        from flashdiffusion.cli import _colored

        result = _colored("test", "blue")
        assert "test" in result

    def test_parser_structure(self):
        from flashdiffusion.cli import main

        import sys

        old = sys.argv
        sys.argv = ["flashdiffusion"]
        with pytest.raises(SystemExit):
            main()
        sys.argv = old


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestDiffusionRegistryExtended:
    def test_len(self):
        from flashdiffusion.registry import Registry

        reg = Registry("test_len")

        @reg.register("X")
        class X:
            pass

        assert len(reg) == 1

    def test_contains(self):
        from flashdiffusion.registry import Registry

        reg = Registry("test_contains")

        @reg.register("Y")
        class Y:
            pass

        assert "Y" in reg
        assert "Z" not in reg


# ---------------------------------------------------------------------------
# Integration: config → create pipeline → (mocked) generate
# ---------------------------------------------------------------------------
class TestDiffusionIntegration:
    def test_scheduler_add_noise_deterministic(self):
        from flashdiffusion.schedulers import DDIMScheduler

        scheduler = DDIMScheduler(num_train_timesteps=1000)
        original = torch.ones(1, 4, 4, 4)
        noise = torch.zeros(1, 4, 4, 4)
        timesteps = torch.tensor([0])
        noisy = scheduler.add_noise(original, noise, timesteps)
        assert torch.allclose(noisy, original, atol=0.01)

    def test_flow_matching_full_loop(self):
        from flashdiffusion.models.sd3 import FlowMatchingScheduler

        s = FlowMatchingScheduler(num_inference_steps=5)
        s.set_timesteps(5)
        sample = torch.randn(1, 4, 8, 8)
        for i, t in enumerate(s.timesteps):
            velocity = torch.randn_like(sample)
            sample = s.step(velocity, t.item(), sample, step_index=i)
        assert sample.shape == (1, 4, 8, 8)
