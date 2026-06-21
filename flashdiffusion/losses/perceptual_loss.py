"""Perceptual losses — LPIPS and VGG feature matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for image quality assessment.

    Computes feature matching loss using intermediate VGG19 layer activations.

    Args:
        layers: VGG layer indices to extract features from.
        normalize: Whether to normalize inputs to VGG expected range.
    """

    def __init__(
        self,
        layers: list = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.layers = layers or [3, 8, 15, 22]
        self._vgg = None

    @property
    def vgg(self):
        if self._vgg is None:
            from torchvision.models import vgg19

            vgg = vgg19(pretrained=True).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self._vgg = vgg
        return self._vgg

    def _extract_features(self, x: torch.Tensor) -> list:
        features = []
        vgg = self.vgg.to(x.device)
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target images.

        Args:
            pred: Predicted image tensor (B, 3, H, W) in [-1, 1].
            target: Target image tensor (B, 3, H, W) in [-1, 1].

        Returns:
            Perceptual loss value.
        """
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
            pred = (pred * 0.5 + 0.5 - mean) / std
            target = (target * 0.5 + 0.5 - mean) / std

        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_features, target_features):
            loss = loss + F.mse_loss(pf, tf.detach())

        return loss / len(pred_features)


class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss wrapper.

    Provides a simplified interface using VGG-based perceptual distance.
    """

    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.perceptual(pred, target)
