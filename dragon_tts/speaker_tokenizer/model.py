"""Standalone speaker tokenizer.

Frozen ECAPA-TDNN + PerceiverResampler + ResidualFSQ + projection.
Inspired by SparkVox BiCodec SpeakerEncoder
(sparkvox/models/codec/BiCodec/modules/speaker_encoder.py), but standalone
and without audio reconstruction loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dragon_tts.modules.ecapa.ecapa_tdnn import ECAPA_TDNN_GLOB_c512
from dragon_tts.modules.fsq.residual_fsq import ResidualFSQ
from dragon_tts.modules.perceiver_encoder import PerceiverResampler


class SpeakerTokenizer(nn.Module):
    """Audio → global speaker tokens.

    Args:
        input_dim: mel feature dim (== ECAPA feat_dim).
        out_dim: x-vector / d-vector dim (== ECAPA embed_dim).
        latent_dim: PerceiverResampler latent + ResidualFSQ working dim.
        token_num: number of global tokens emitted per clip.
        fsq_levels: per-level codebook size for FSQ; codebook_size = prod(levels).
        fsq_num_quantizers: number of residual quantizer layers.
        perceiver_depth: number of cross-attn + FF blocks in PerceiverResampler.
        ecapa_channels: internal channels of ECAPA SE-Res2 blocks (512 by default).
    """

    def __init__(
        self,
        input_dim: int = 128,
        out_dim: int = 1024,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: Optional[List[int]] = None,
        fsq_num_quantizers: int = 1,
        perceiver_depth: int = 2,
        ecapa_channels: int = 512,
    ):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [4, 4, 4, 4, 4, 4]

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.token_num = token_num
        self.fsq_levels = list(fsq_levels)
        self.fsq_num_quantizers = fsq_num_quantizers

        assert ecapa_channels == 512, (
            "Plan vendors ECAPA_TDNN_GLOB_c512; adjust dim_context if you change "
            "this (it is hard-coded to channels * 3 = 1536 below)."
        )

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(
            feat_dim=input_dim, embed_dim=out_dim
        )
        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim,
            dim_context=ecapa_channels * 3,
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
        self.project = nn.Linear(latent_dim * token_num, out_dim)

        self._ecapa_frozen = False

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    def freeze_ecapa(self) -> None:
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False
        self.speaker_encoder.eval()
        self._ecapa_frozen = True

    def load_ecapa_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """Load weights into the ECAPA submodule and freeze it."""
        missing, unexpected = self.speaker_encoder.load_state_dict(
            state_dict, strict=strict
        )
        if missing:
            print(f"[SpeakerTokenizer] ECAPA missing keys: {missing}")
        if unexpected:
            print(f"[SpeakerTokenizer] ECAPA unexpected keys: {unexpected}")
        self.freeze_ecapa()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._ecapa_frozen:
            # Keep ECAPA in eval mode (BatchNorm running stats, etc.)
            self.speaker_encoder.eval()
        return self

    def _encode_features(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ECAPA forward. If ECAPA is frozen, no grad."""
        if self._ecapa_frozen:
            with torch.no_grad():
                x_vector, features = self.speaker_encoder(mels, True)
        else:
            x_vector, features = self.speaker_encoder(mels, True)
        return x_vector, features

    def forward(
        self, mels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            mels: (B, D_mel, T) mel-spectrogram (channels-first).

        Returns:
            x_vector: (B, out_dim) raw ECAPA embedding.
            d_vector: (B, out_dim) reconstructed embedding from quantized tokens.
            indices:  (B, num_quantizers, token_num) global token indices
                (channels-first convention, khớp với SparkVox SpeakerEncoder).
        """
        mels = mels.transpose(1, 2)  # ECAPA expects (B, T, F)
        x_vector, features = self._encode_features(mels)

        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(x)
        d_vector = self.project(zq.reshape(zq.shape[0], -1))

        return x_vector, d_vector, indices

    @torch.no_grad()
    def tokenize(self, mels: torch.Tensor) -> torch.Tensor:
        """Return indices only, skipping the projection branch."""
        mels = mels.transpose(1, 2)
        _, features = self._encode_features(mels)
        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        _, indices = self.quantizer(x)
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
        input_dim=128,
        out_dim=1024,
        latent_dim=128,
        token_num=32,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=1,
    )
    model.freeze_ecapa()
    mel = torch.randn(2, 128, 200)
    x_vec, d_vec, indices = model(mel)
    print("x_vector:", x_vec.shape)
    print("d_vector:", d_vec.shape)
    print("indices :", indices.shape)
    print("codebook_size:", model.codebook_size)

    idx = model.tokenize(mel)
    d = model.detokenize(idx)
    print("round-trip d_vector:", d.shape)
