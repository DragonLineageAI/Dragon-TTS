"""Standalone speaker tokenizer.

Frozen speaker-embedding backbone + PerceiverResampler + ResidualFSQ +
projection. Inspired by SparkVox BiCodec SpeakerEncoder
(sparkvox/models/codec/BiCodec/modules/speaker_encoder.py), but standalone and
without an audio reconstruction loss.

The speaker backbone is selectable via the ``encoder`` config block (see
``dragon_tts/speaker_tokenizer/backbones.py``):
  - ``ecapa`` — frozen ECAPA-TDNN on LINEAR mel (the original design).
  - ``wavlm`` — frozen ``microsoft/wavlm-base-plus-sv`` on raw 16 kHz waveform.
Everything downstream of the backbone is encoder-agnostic; ``context_dim`` and
``out_dim`` are derived from the chosen backbone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dragon_tts.speaker_tokenizer.backbones import SpeakerBackbone, build_backbone
from dragon_tts.modules.fsq.residual_fsq import ResidualFSQ
from dragon_tts.modules.perceiver_encoder import PerceiverResampler


class SpeakerTokenizer(nn.Module):
    """Audio → global speaker tokens.

    Args:
        encoder: backbone config dict, e.g.
            ``{"type": "ecapa", "feat_dim": 128, "embed_dim": 1024, "channels": 512}``
            or ``{"type": "wavlm", "pretrained": "microsoft/wavlm-base-plus-sv"}``.
            If ``None``, falls back to the legacy ECAPA kwargs below.
        latent_dim: PerceiverResampler latent + ResidualFSQ working dim.
        token_num: number of global tokens emitted per clip.
        fsq_levels: per-level codebook size for FSQ; codebook_size = prod(levels).
        fsq_num_quantizers: number of residual quantizer layers.
        perceiver_depth: number of cross-attn + FF blocks in PerceiverResampler.
        input_dim / out_dim / ecapa_channels: legacy ECAPA kwargs, used only when
            ``encoder`` is None (kept for backward compatibility).
    """

    def __init__(
        self,
        encoder: Optional[Dict[str, Any]] = None,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: Optional[List[int]] = None,
        fsq_num_quantizers: int = 1,
        perceiver_depth: int = 2,
        # ---- legacy ECAPA kwargs (used only if `encoder` is None) ----
        input_dim: int = 128,
        out_dim: int = 1024,
        ecapa_channels: int = 512,
    ):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [4, 4, 4, 4, 4, 4]

        if encoder is None:
            # Backward-compatible default: frozen ECAPA on mel.
            encoder = {
                "type": "ecapa",
                "feat_dim": input_dim,
                "embed_dim": out_dim,
                "channels": ecapa_channels,
            }

        self.latent_dim = latent_dim
        self.token_num = token_num
        self.fsq_levels = list(fsq_levels)
        self.fsq_num_quantizers = fsq_num_quantizers

        self.speaker_encoder: SpeakerBackbone = build_backbone(encoder)
        dim_context = self.speaker_encoder.context_dim
        self.out_dim = self.speaker_encoder.embed_dim
        self.input_dim = input_dim

        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim,
            dim_context=dim_context,
            num_latents=token_num,
            depth=perceiver_depth,
        )
        self.quantizer = ResidualFSQ(
            levels=self.fsq_levels,
            num_quantizers=fsq_num_quantizers,
            dim=latent_dim,
            is_channel_first=True,
            quantize_dropout=False,
        )
        self.project = nn.Linear(latent_dim * token_num, self.out_dim)

        self._encoder_frozen = False

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    @property
    def input_kind(self) -> str:
        """"mel" or "waveform" — what the configured backbone expects."""
        return self.speaker_encoder.input_kind

    # ------------------------------------------------------------------
    # Freeze machinery (encoder-agnostic; ECAPA aliases kept for back-compat)
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False
        self.speaker_encoder.eval()
        self._encoder_frozen = True

    def load_encoder_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Load weights into the backbone's encoder submodule and freeze it.

        Only meaningful for backbones that consume an external state_dict (ECAPA).
        WavLM loads its weights via ``from_pretrained`` at construction time.
        """
        loader = getattr(self.speaker_encoder, "load_encoder_state_dict", None)
        if loader is None:
            raise AttributeError(
                f"{type(self.speaker_encoder).__name__} does not support loading an "
                "external state_dict (weights come from from_pretrained)."
            )
        missing, unexpected = loader(state_dict, strict=strict)
        if missing:
            print(f"[SpeakerTokenizer] encoder missing keys: {missing}")
        if unexpected:
            print(f"[SpeakerTokenizer] encoder unexpected keys: {unexpected}")
        self.freeze_encoder()

    # Backward-compatible aliases.
    freeze_ecapa = freeze_encoder
    load_ecapa_state_dict = load_encoder_state_dict

    @property
    def _ecapa_frozen(self) -> bool:  # back-compat read access
        return self._encoder_frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            # Keep the backbone in eval mode: ECAPA BatchNorm running stats and
            # WavLM dropout/LayerNorm must stay deterministic.
            self.speaker_encoder.eval()
        return self

    def _encode_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone. If frozen, no grad."""
        if self._encoder_frozen:
            with torch.no_grad():
                x_vector, features = self.speaker_encoder(x)
        else:
            x_vector, features = self.speaker_encoder(x)
        return x_vector, features

    # ------------------------------------------------------------------
    # Forward / tokenize / detokenize
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            x: encoder input — mel ``(B, n_mels, T)`` for ecapa, or raw waveform
               ``(B, T)`` for wavlm (see ``self.input_kind``).

        Returns:
            x_vector: (B, out_dim) raw backbone embedding (reconstruction target).
            d_vector: (B, out_dim) reconstructed embedding from quantized tokens.
            indices:  (B, num_quantizers, token_num) global token indices
                (channels-first convention, khớp với SparkVox SpeakerEncoder).
        """
        x_vector, features = self._encode_features(x)

        h = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(h)
        d_vector = self.project(zq.reshape(zq.shape[0], -1))

        return x_vector, d_vector, indices

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Return indices only, skipping the projection branch."""
        _, features = self._encode_features(x)
        h = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        _, indices = self.quantizer(h)
        return indices

    @torch.no_grad()
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """Recover d_vector from indices."""
        zq = self.quantizer.get_output_from_indices(indices.transpose(1, 2)).transpose(
            1, 2
        )
        d_vector = self.project(zq.reshape(zq.shape[0], -1))
        return d_vector


if __name__ == "__main__":
    model = SpeakerTokenizer(
        encoder={"type": "ecapa", "feat_dim": 128, "embed_dim": 1024, "channels": 512},
        latent_dim=128,
        token_num=32,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=1,
    )
    model.freeze_encoder()
    mel = torch.randn(2, 128, 200)
    x_vec, d_vec, indices = model(mel)
    print("x_vector:", x_vec.shape)
    print("d_vector:", d_vec.shape)
    print("indices :", indices.shape)
    print("codebook_size:", model.codebook_size)

    idx = model.tokenize(mel)
    d = model.detokenize(idx)
    print("round-trip d_vector:", d.shape)
