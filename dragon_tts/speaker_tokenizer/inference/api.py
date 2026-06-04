"""Public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

from dragon_tts.speaker_tokenizer.data.mel import LogMelSpectrogram, MelConfig
from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer
from dragon_tts.speaker_tokenizer.training.lit_module import SpeakerTokenizerLit


AudioInput = Union[str, Path, torch.Tensor]


class SpeakerTokenizerPipeline:
    """Wrap `SpeakerTokenizer` with audio I/O + mel transform.

    Use cases:
      - `tokenize(audio)` → indices ``(B, token_num, num_quantizers)`` ready to
        be consumed by a downstream TTS model.
      - `detokenize(indices)` → reconstructed d_vector ``(B, out_dim)``.
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        device: str = "cuda",
        mel_cfg: Optional[MelConfig] = None,
    ):
        self.device = torch.device(device)
        lit = SpeakerTokenizerLit.load_from_checkpoint(
            str(ckpt_path), map_location=self.device
        )
        lit.eval()
        lit.to(self.device)
        self.model: SpeakerTokenizer = lit.model

        if mel_cfg is None:
            hp_mel = lit.hparams.get("model_cfg", {})  # not directly storing mel
            mel_cfg = MelConfig()  # defaults are 24 kHz / 128 mels — matches BiCodec
        self.mel_cfg = mel_cfg
        self.mel = LogMelSpectrogram(mel_cfg).to(self.device)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _load_wav(self, path: Union[str, Path]) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.mel_cfg.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.mel_cfg.sample_rate
            )
        return wav.to(self.device)  # (1, T)

    def _to_mel(self, audio: AudioInput) -> torch.Tensor:
        if isinstance(audio, (str, Path)):
            wav = self._load_wav(audio)  # (1, T)
        else:
            wav = audio.to(self.device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        mel = self.mel(wav)  # (B, n_mels, T)
        return mel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tokenize(self, audio: AudioInput) -> torch.Tensor:
        mel = self._to_mel(audio)
        return self.model.tokenize(mel)

    @torch.no_grad()
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.model.detokenize(indices.to(self.device))

    @torch.no_grad()
    def encode(self, audio: AudioInput) -> dict:
        """Return all forward outputs: x_vector, d_vector, indices."""
        mel = self._to_mel(audio)
        x_vec, d_vec, indices = self.model(mel)
        return {"x_vector": x_vec, "d_vector": d_vec, "indices": indices}

    @torch.no_grad()
    def tokenize_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model.tokenize(mel.to(self.device))
