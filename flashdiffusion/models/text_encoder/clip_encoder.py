"""CLIP text encoder wrapper for diffusion model conditioning."""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CLIPTextEncoderWrapper(nn.Module):
    """Wrapper around HuggingFace CLIPTextModel for prompt encoding.

    Args:
        model_id: HuggingFace model ID.
        subfolder: Subfolder within the model repo (default: "text_encoder").
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "text_encoder",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.subfolder = subfolder
        self.torch_dtype = torch_dtype or torch.float32
        self._encoder = None
        self._tokenizer = None

    @property
    def encoder(self):
        if self._encoder is None:
            self._load()
        return self._encoder

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def _load(self):
        from transformers import CLIPTextModel, CLIPTokenizer

        self._encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder=self.subfolder,
            torch_dtype=self.torch_dtype,
        )
        self._tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
        )
        logger.info("CLIP text encoder loaded from %s", self.model_id)

    @torch.no_grad()
    def encode(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode text prompt(s) into embeddings.

        Args:
            prompt: Text prompt or list of prompts.
            max_length: Maximum token length.
            device: Target device for output tensors.

        Returns:
            Text embeddings tensor (B, seq_len, hidden_dim).
        """
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens.input_ids
        if device is not None:
            input_ids = input_ids.to(device)

        output = self.encoder(input_ids)
        return output.last_hidden_state

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids).last_hidden_state
