"""Mel-spectrogram transform khớp với cấu hình BiCodec (Spark-TTS-0.5B).

QUAN TRỌNG: ECAPA trong BiCodec được train trên mel **tuyến tính** (power=1,
KHÔNG lấy log). SparkVox `Generator.init_mel_transformer` đưa thẳng output của
`torchaudio MelSpectrogram` vào speaker encoder mà không có log/clamp. Bón log-mel
vào sẽ làm lệch phân phối của các BatchNorm (frozen) trong ECAPA → x_vector nổ
(norm ~1e6) và MSE không học được.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio.transforms as TT


@dataclass
class MelConfig:
    # Khớp checkpoints/Spark-TTS-0.5B/BiCodec/config.yaml (mel_params).
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 640
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 10.0
    f_max: Optional[float] = None  # None ⇒ sample_rate / 2


class MelSpectrogramFeature(nn.Module):
    """Linear mel-spectrogram; sao chép `Generator.init_mel_transformer` của BiCodec."""

    def __init__(self, cfg: MelConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = TT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Args:
            wav: (B, T) hoặc (B, 1, T) hoặc (T,).

        Returns:
            linear mel: (B, n_mels, T_frames). KHÔNG lấy log (xem docstring module).
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return self.mel(wav)
