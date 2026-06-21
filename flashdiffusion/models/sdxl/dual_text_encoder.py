"""SDXL dual text encoder — CLIP-L + OpenCLIP-G with attention pooling."""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SDXLDualTextEncoder(nn.Module):
    """Dual text encoder for SDXL: CLIP-L (text_encoder) + OpenCLIP-G (text_encoder_2).

    Produces concatenated hidden states from both encoders, plus pooled
    output from OpenCLIP-G for conditioning the time embedding.

    Args:
        model_id: SDXL model ID.
        torch_dtype: Data type for model weights.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.torch_dtype = torch_dtype or torch.float16
        self._encoder_1 = None
        self._encoder_2 = None
        self._tokenizer_1 = None
        self._tokenizer_2 = None

    def _load(self):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self._encoder_1 = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
        )
        self._encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_id,
            subfolder="text_encoder_2",
            torch_dtype=self.torch_dtype,
        )
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
        )
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer_2",
        )
        logger.info("SDXL dual text encoders loaded from %s", self.model_id)

    @property
    def encoder_1(self):
        if self._encoder_1 is None:
            self._load()
        return self._encoder_1

    @property
    def encoder_2(self):
        if self._encoder_2 is None:
            self._load()
        return self._encoder_2

    @property
    def tokenizer_1(self):
        if self._tokenizer_1 is None:
            self._load()
        return self._tokenizer_1

    @property
    def tokenizer_2(self):
        if self._tokenizer_2 is None:
            self._load()
        return self._tokenizer_2

    @torch.no_grad()
    def encode(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts using both CLIP text encoders.

        Args:
            prompt: Text prompt(s).
            device: Target device.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds).
            prompt_embeds: Concatenated hidden states (B, seq_len, 2048).
            pooled_prompt_embeds: Pooled output from encoder_2 (B, 1280).
        """
        tokens_1 = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        if device is not None:
            tokens_1 = {k: v.to(device) for k, v in tokens_1.items()}
            tokens_2 = {k: v.to(device) for k, v in tokens_2.items()}

        output_1 = self.encoder_1(tokens_1["input_ids"], output_hidden_states=True)
        output_2 = self.encoder_2(tokens_2["input_ids"], output_hidden_states=True)

        hidden_1 = output_1.hidden_states[-2]
        hidden_2 = output_2.hidden_states[-2]

        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        pooled_prompt_embeds = output_2.text_embeds

        return prompt_embeds, pooled_prompt_embeds

    def forward(
        self,
        input_ids_1: torch.Tensor,
        input_ids_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_1 = self.encoder_1(input_ids_1, output_hidden_states=True)
        output_2 = self.encoder_2(input_ids_2, output_hidden_states=True)

        hidden_1 = output_1.hidden_states[-2]
        hidden_2 = output_2.hidden_states[-2]

        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        pooled_prompt_embeds = output_2.text_embeds

        return prompt_embeds, pooled_prompt_embeds
