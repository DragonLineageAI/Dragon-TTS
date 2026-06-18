"""Mel-spectrogram transform khớp với cấu hình BiCodec (Spark-TTS-0.5B).

QUAN TRỌNG: ECAPA trong BiCodec được train trên mel **tuyến tính** (power=1,
KHÔNG lấy log). SparkVox `Generator.init_mel_transformer` đưa thẳng output của
MelSpectrogram vào speaker encoder mà không có log/clamp. Bón log-mel
vào sẽ làm lệch phân phối của các BatchNorm (frozen) trong ECAPA → x_vector nổ
(norm ~1e6) và MSE không học được.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


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
    """Linear mel-spectrogram; sao chép `Generator.init_mel_transformer` của BiCodec.

    Uses librosa mel basis + ``torch.stft`` (no torchaudio dependency).
    """

    def __init__(self, cfg: MelConfig):
        super().__init__()
        self.cfg = cfg
        from librosa.filters import mel as librosa_mel_fn

        fmax = cfg.f_max if cfg.f_max is not None else cfg.sample_rate / 2.0
        mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sr=cfg.sample_rate,
                n_fft=cfg.n_fft,
                n_mels=cfg.n_mels,
                fmin=cfg.f_min,
                fmax=fmax,
            )
        ).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer(
            "hann_window", torch.hann_window(cfg.win_length)
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

        cfg = self.cfg
        # Reflect-pad to center the STFT windows, matching torchaudio behaviour.
        pad_len = (cfg.n_fft - cfg.hop_length) // 2
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (pad_len, pad_len), mode="reflect").squeeze(1)

        spec = torch.stft(
            wav,
            cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=self.hann_window,
            center=False,
            return_complex=True,
        )
        spec = torch.abs(spec)  # magnitude: (B, n_fft//2+1, T_frames)
        mel = torch.matmul(self.mel_basis, spec)  # (B, n_mels, T_frames)
        return mel  # LINEAR — no log
