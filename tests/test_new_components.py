"""Tests for new FlashDiffusion P0 components — SDXL, SD3, adapters, video, upscaling."""

import torch

from flashdiffusion.models.sd3 import MMDiTBlock, FlowMatchingScheduler
from flashdiffusion.models.sdxl.sdxl_unet import SDXLUNetWrapper
from flashdiffusion.adapters.ip_adapter import DecoupledCrossAttention
from flashdiffusion.adapters.instant_id import IdentityNet
from flashdiffusion.video.animatediff import TemporalAttention, AnimateDiffMotionModule
from flashdiffusion.upscaling.esrgan import DenseBlock, RRDB, RRDBNet


class TestMMDiT:
    def test_block_forward(self):
        block = MMDiTBlock(hidden_size=64, num_heads=4, mlp_ratio=4.0)
        img = torch.randn(2, 16, 64)
        txt = torch.randn(2, 8, 64)
        t_emb = torch.randn(2, 64)

        img_out, txt_out = block(img, txt, t_emb)
        assert img_out.shape == (2, 16, 64)
        assert txt_out.shape == (2, 8, 64)

    def test_mmdit_gradient_flow(self):
        block = MMDiTBlock(hidden_size=64, num_heads=4)
        img = torch.randn(1, 4, 64, requires_grad=True)
        txt = torch.randn(1, 4, 64, requires_grad=True)
        t_emb = torch.randn(1, 64)

        img_out, txt_out = block(img, txt, t_emb)
        loss = img_out.sum() + txt_out.sum()
        loss.backward()
        assert img.grad is not None


class TestFlowMatching:
    def test_set_timesteps(self):
        scheduler = FlowMatchingScheduler(num_inference_steps=28, shift=3.0)
        scheduler.set_timesteps(28)
        assert len(scheduler.timesteps) == 28

    def test_step(self):
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(10)
        sample = torch.randn(1, 4, 64, 64)
        velocity = torch.randn(1, 4, 64, 64)
        result = scheduler.step(velocity, 0.5, sample, step_index=0)
        assert result.shape == sample.shape

    def test_add_noise(self):
        scheduler = FlowMatchingScheduler()
        original = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(original)
        noisy = scheduler.add_noise(original, noise, sigma=0.5)
        assert noisy.shape == original.shape


class TestSDXLUNet:
    def test_time_ids(self):
        time_ids = SDXLUNetWrapper.compute_time_ids(
            original_size=(1024, 1024),
            crop_coords=(0, 0),
            target_size=(1024, 1024),
            device="cpu",
        )
        assert time_ids.shape == (1, 6)
        assert time_ids[0, 0].item() == 1024


class TestDecoupledCrossAttention:
    def test_forward(self):
        attn = DecoupledCrossAttention(hidden_size=64, cross_attention_dim=32, num_heads=4)
        query = torch.randn(2, 16, 64)
        image_embeds = torch.randn(2, 4, 32)
        out = attn(query, image_embeds)
        assert out.shape == (2, 16, 64)


class TestIdentityNet:
    def test_forward(self):
        net = IdentityNet(face_embed_dim=512, cross_attention_dim=768, num_tokens=16)
        embedding = torch.randn(2, 512)
        tokens = net(embedding)
        assert tokens.shape == (2, 16, 768)


class TestTemporalAttention:
    def test_forward(self):
        attn = TemporalAttention(channels=64, num_heads=4, num_frames=8)
        x = torch.randn(16, 64, 4, 4)  # B*T=16, C=64, H=W=4
        out = attn(x, num_frames=8)
        assert out.shape == (16, 64, 4, 4)


class TestAnimateDiffMotionModule:
    def test_module_creation(self):
        module = AnimateDiffMotionModule(
            channels_list=[64, 128], num_frames=8, num_heads=4,
        )
        assert "down_0" in module.temporal_attentions
        assert "mid" in module.temporal_attentions

    def test_apply_temporal(self):
        module = AnimateDiffMotionModule(
            channels_list=[64], num_frames=4, num_heads=4,
        )
        x = torch.randn(8, 64, 4, 4)
        out = module.apply_temporal_attention(x, "down_0")
        assert out.shape == x.shape


class TestESRGAN:
    def test_dense_block(self):
        block = DenseBlock(channels=32, growth_rate=16)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == (1, 32, 8, 8)

    def test_rrdb(self):
        rrdb = RRDB(channels=32, growth_rate=16)
        x = torch.randn(1, 32, 8, 8)
        out = rrdb(x)
        assert out.shape == (1, 32, 8, 8)

    def test_rrdbnet_forward(self):
        model = RRDBNet(
            in_channels=3, out_channels=3, num_features=32,
            num_blocks=2, growth_rate=16, scale=4,
        )
        x = torch.randn(1, 3, 16, 16)
        out = model(x)
        assert out.shape == (1, 3, 64, 64)

    def test_rrdbnet_2x(self):
        model = RRDBNet(
            in_channels=3, out_channels=3, num_features=32,
            num_blocks=2, growth_rate=16, scale=2,
        )
        x = torch.randn(1, 3, 16, 16)
        out = model(x)
        assert out.shape == (1, 3, 32, 32)
