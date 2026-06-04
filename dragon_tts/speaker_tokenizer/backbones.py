"""Speaker-embedding backbones for the SpeakerTokenizer.

A backbone turns audio into:
  - ``x_vector`` ``(B, embed_dim)`` — the pooled speaker embedding that the
    tokenizer learns to reconstruct (the training target).
  - ``features`` ``(B, context_dim, T)`` — frame-level features that feed the
    PerceiverResampler as cross-attention context.

Two backbones are provided, selectable via config:
  - ``ecapa``  — frozen ECAPA-TDNN on LINEAR mel (the original design).
  - ``wavlm``  — frozen ``microsoft/wavlm-base-plus-sv`` on raw 16 kHz waveform.

Everything downstream (PerceiverResampler → ResidualFSQ → projection) is
encoder-agnostic and only needs ``context_dim`` / ``embed_dim`` to be sized
correctly, plus ``input_kind`` so the data pipeline knows whether to feed mel
or waveform.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from dragon_tts.modules.ecapa.ecapa_tdnn import ECAPA_TDNN_GLOB_c512


class SpeakerBackbone(nn.Module):
    """Common interface for speaker-embedding backbones."""

    #: "mel" → expects mel-spectrogram (B, n_mels, T); "waveform" → raw (B, T).
    input_kind: str = "mel"

    @property
    def context_dim(self) -> int:
        """Channel dim ``C`` of ``features`` (PerceiverResampler dim_context)."""
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        """Dim ``E`` of ``x_vector`` (== tokenizer out_dim / projection output)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_vector (B, E), features (B, C, T))``."""
        raise NotImplementedError


class EcapaBackbone(SpeakerBackbone):
    """Frozen ECAPA-TDNN on LINEAR mel features.

    Numerically identical to the original inline wiring: same module, same
    ``(B, n_mels, T) -> (B, T, n_mels)`` transpose before ECAPA, same
    ``return_latent=True`` call.
    """

    input_kind = "mel"

    def __init__(
        self,
        feat_dim: int = 128,
        embed_dim: int = 1024,
        channels: int = 512,
    ):
        super().__init__()
        assert channels == 512, (
            "Only ECAPA_TDNN_GLOB_c512 is vendored; context_dim is channels*3 "
            "= 1536. Adjust if you vendor another ECAPA variant."
        )
        self._channels = channels
        self._embed_dim = embed_dim
        self.encoder = ECAPA_TDNN_GLOB_c512(feat_dim=feat_dim, embed_dim=embed_dim)

    @property
    def context_dim(self) -> int:
        return self._channels * 3  # cat(out2, out3, out4) → 1536

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def load_encoder_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ):
        """Load a standalone ECAPA state_dict (bare keys) into the encoder."""
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def forward(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = mels.transpose(1, 2)  # ECAPA expects (B, T, F)
        x_vector, features = self.encoder(x, True)  # (B, embed_dim), (B, 1536, T)
        return x_vector, features


class WavLMBackbone(SpeakerBackbone):
    """Frozen ``WavLMForXVector`` (microsoft/wavlm-base-plus-sv) on raw waveform.

    The pooled xvector ``embeddings`` (512-d) is the reconstruction target; the
    base encoder's last hidden state (768-ch) is the PerceiverResampler context.
    Both come from a single forward (``output_hidden_states=True``).
    """

    input_kind = "waveform"

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus-sv",
        do_normalize: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        # Imported lazily so the rest of the package works without transformers.
        from transformers import WavLMForXVector

        self.model = WavLMForXVector.from_pretrained(pretrained)
        self.do_normalize = do_normalize
        self._context_dim = int(self.model.config.hidden_size)  # 768
        self._embed_dim = int(self.model.config.xvector_output_dim)  # 512
        # `freeze` here only records intent; the SpeakerTokenizer owns the actual
        # requires_grad / eval state via freeze_encoder(). Kept for clarity.
        self._freeze = freeze

    @property
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _normalize(self, wav: torch.Tensor) -> torch.Tensor:
        # Per-utterance zero-mean/unit-var, matching the SV feature extractor.
        # Done in fp32 with an epsilon guard for silent/padded crops.
        wav = wav.float()
        mean = wav.mean(dim=-1, keepdim=True)
        std = wav.std(dim=-1, keepdim=True)
        return (wav - mean) / (std + 1e-7)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if self.do_normalize:
            wav = self._normalize(wav)
        out = self.model(input_values=wav, output_hidden_states=True)
        x_vector = out.embeddings  # (B, 512)
        features = out.hidden_states[-1].transpose(1, 2)  # (B, 768, T)
        return x_vector, features


def build_backbone(encoder_cfg: Dict[str, Any]) -> SpeakerBackbone:
    """Construct a backbone from a config dict with a ``type`` key.

    ``type="ecapa"``  → ``EcapaBackbone(feat_dim, embed_dim, channels)``
    ``type="wavlm"``  → ``WavLMBackbone(pretrained, do_normalize, freeze)``

    Extra keys not consumed by the backbone (e.g. ``ckpt``) are ignored here;
    the SpeakerTokenizer / LightningModule handle checkpoint loading.
    """
    cfg = dict(encoder_cfg)
    enc_type = cfg.pop("type")
    if enc_type == "ecapa":
        return EcapaBackbone(
            feat_dim=cfg.get("feat_dim", 128),
            embed_dim=cfg.get("embed_dim", 1024),
            channels=cfg.get("channels", 512),
        )
    if enc_type == "wavlm":
        return WavLMBackbone(
            pretrained=cfg.get("pretrained", "microsoft/wavlm-base-plus-sv"),
            do_normalize=cfg.get("do_normalize", True),
            freeze=cfg.get("freeze", True),
        )
    raise ValueError(f"Unknown encoder type: {enc_type!r} (expected 'ecapa' or 'wavlm')")
