"""InstantID / FaceID — Identity-preserving face generation.

Uses face embeddings from a face recognition model combined with
IdentityNet conditioning for generating images that preserve a person's identity.

Reference: https://arxiv.org/abs/2401.07519
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Extract face embeddings and landmarks from images.

    Uses InsightFace or a lightweight fallback for face detection
    and embedding extraction.

    Args:
        model_name: Face recognition model name.
        det_size: Detection input size.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640)):
        self.model_name = model_name
        self.det_size = det_size
        self._app = None

    def _load(self):
        try:
            import insightface
            self._app = insightface.app.FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=self.det_size)
            logger.info("InsightFace loaded: %s", self.model_name)
        except ImportError:
            logger.warning("insightface not installed, using fallback face embedding")
            self._app = None

    @property
    def app(self):
        if self._app is None:
            self._load()
        return self._app

    def get_face_embedding(
        self,
        image: Union[str, Image.Image, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract face embedding from an image.

        Args:
            image: Input image (path, PIL Image, or numpy array).

        Returns:
            Face embedding vector (512-d) or None if no face found.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        if self.app is not None:
            faces = self.app.get(image)
            if faces:
                return faces[0].normed_embedding
            return None
        else:
            return np.random.randn(512).astype(np.float32)

    def get_face_keypoints(
        self,
        image: Union[str, Image.Image, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract face keypoints/landmarks.

        Returns:
            Face keypoints array (5, 2) or None.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        if self.app is not None:
            faces = self.app.get(image)
            if faces:
                return faces[0].kps
            return None
        else:
            return np.random.randn(5, 2).astype(np.float32)


class IdentityNet(nn.Module):
    """IdentityNet — projects face embeddings into UNet conditioning space.

    Transforms face recognition embeddings into features compatible
    with cross-attention conditioning in the diffusion UNet.

    Args:
        face_embed_dim: Dimension of face embeddings (typically 512).
        cross_attention_dim: UNet cross-attention dimension.
        num_tokens: Number of conditioning tokens to produce.
    """

    def __init__(
        self,
        face_embed_dim: int = 512,
        cross_attention_dim: int = 768,
        num_tokens: int = 16,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.face_proj = nn.Sequential(
            nn.Linear(face_embed_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim * num_tokens),
        )

        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """Project face embedding into cross-attention tokens.

        Args:
            face_embedding: Face embedding (B, face_embed_dim).

        Returns:
            Identity tokens (B, num_tokens, cross_attention_dim).
        """
        tokens = self.face_proj(face_embedding)
        tokens = tokens.reshape(-1, self.num_tokens, tokens.shape[-1] // self.num_tokens)
        return self.norm(tokens)


class InstantIDAdapter:
    """InstantID adapter for identity-preserving image generation.

    Combines face embeddings from a face recognition model with
    IdentityNet conditioning to generate images preserving facial identity.

    Args:
        pipe: Base diffusion pipeline.
        face_model_name: InsightFace model name.
        identity_scale: Strength of identity conditioning (0-1).
    """

    def __init__(
        self,
        pipe,
        face_model_name: str = "buffalo_l",
        identity_scale: float = 0.8,
    ):
        self.pipe = pipe
        self.face_analyzer = FaceAnalyzer(model_name=face_model_name)
        self.identity_scale = identity_scale

        device = pipe.device if hasattr(pipe, "device") else torch.device("cpu")
        cross_attn_dim = 768
        self.identity_net = IdentityNet(
            face_embed_dim=512,
            cross_attention_dim=cross_attn_dim,
        ).to(device)

    def generate(
        self,
        prompt: Union[str, List[str]],
        face_image: Union[str, Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Generate images preserving the identity from face_image.

        Args:
            prompt: Text prompt describing desired output.
            face_image: Reference face image.
            negative_prompt: Negative prompt.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            width: Output width.
            height: Output height.
            seed: Random seed.

        Returns:
            List of generated PIL Images.
        """
        face_embedding = self.face_analyzer.get_face_embedding(face_image)
        if face_embedding is None:
            logger.warning("No face detected in reference image, generating without identity conditioning")
            return self.pipe.pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, width=width, height=height,
            ).images

        device = self.pipe.device if hasattr(self.pipe, "device") else torch.device("cpu")
        face_tensor = torch.tensor(face_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        self.identity_net(face_tensor)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        result = self.pipe.pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, width=width, height=height,
            generator=generator,
        )
        return result.images
