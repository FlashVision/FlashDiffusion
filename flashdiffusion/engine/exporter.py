"""FlashDiffusion Exporter — export models to ONNX and TensorRT."""

import os
import logging

import torch

logger = logging.getLogger(__name__)


class Exporter:
    """Export diffusion model components to ONNX format.

    Example::

        from flashdiffusion import Exporter

        exporter = Exporter(model_id="runwayml/stable-diffusion-v1-5")
        exporter.export(output="unet.onnx")
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        component: str = "unet",
    ):
        self.model_id = model_id
        self.component = component

    def export(
        self,
        output: str = "unet.onnx",
        opset_version: int = 14,
    ) -> str:
        """Export model component to ONNX format.

        Args:
            output: Output path for the ONNX file.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported file.
        """
        return self.export_onnx(output_path=output, opset_version=opset_version)

    def export_onnx(
        self,
        output_path: str = "unet.onnx",
        opset_version: int = 14,
    ) -> str:
        """Export UNet to ONNX format.

        Args:
            output_path: Output file path.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported ONNX file.
        """
        from diffusers import UNet2DConditionModel

        logger.info("Loading UNet for export: %s", self.model_id)
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=torch.float32,
        )
        unet.eval()

        sample_size = unet.config.sample_size
        in_channels = unet.config.in_channels
        cross_attention_dim = unet.config.cross_attention_dim

        dummy_latent = torch.randn(1, in_channels, sample_size, sample_size)
        dummy_timestep = torch.tensor([1])
        dummy_encoder_hidden = torch.randn(1, 77, cross_attention_dim)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        torch.onnx.export(
            unet,
            (dummy_latent, dummy_timestep, dummy_encoder_hidden),
            output_path,
            opset_version=opset_version,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "sample": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "noise_pred": {0: "batch"},
            },
        )

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("ONNX exported: %s (%.2f MB)", output_path, file_size)
        return output_path
