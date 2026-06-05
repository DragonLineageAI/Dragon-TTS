"""Abstract base class for audio codecs used in the LLM-TTS pipeline.

Both ``SnacCodec`` and ``NeucodecCodec`` implement this interface so that all
upstream / downstream code (data building, inference, streaming) can be
codec-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import torch


class AudioCodec(ABC):
    """Unified interface for neural audio codecs (SNAC, NeuCodec, …)."""

    # ── class-level codec properties (override in subclasses) ──────────
    sample_rate: int  # input sample rate the codec expects
    output_sample_rate: int  # sample rate of decoded audio
    slots_per_frame: int  # tokens emitted per frame (7 for SNAC, 1 for NeuCodec)
    codebook_size: int  # code values per slot (4096 for SNAC, 65536 for NeuCodec)

    @property
    def num_audio_tokens(self) -> int:
        """Total number of unique audio token ids = slots × codebook_size."""
        return self.slots_per_frame * self.codebook_size

    # ── encode ─────────────────────────────────────────────────────────
    @abstractmethod
    def encode(self, wav: torch.Tensor) -> List[str]:
        """Encode a single waveform into a flat list of ``<|audio_*|>`` tokens.

        ``wav`` should be at the codec's expected ``sample_rate``.
        """
        ...

    @abstractmethod
    def batch_encode(self, wavs: List[torch.Tensor]) -> List[List[str]]:
        """Batch-encode multiple waveforms.

        Each ``wavs[i]`` is a 1-D tensor at ``self.sample_rate``.
        """
        ...

    # ── decode ─────────────────────────────────────────────────────────
    @abstractmethod
    def decode(self, audio_tokens: Sequence[str]) -> torch.Tensor:
        """Decode ``<|audio_*|>`` token strings back to a waveform.

        Returns a 1-D CPU tensor at ``self.output_sample_rate``.
        """
        ...

    @abstractmethod
    def decode_pairs(self, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Decode parsed ``(slot, value)`` pairs to a waveform."""
        ...
