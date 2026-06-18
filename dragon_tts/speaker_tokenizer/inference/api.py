"""Public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

from dragon_tts.speaker_tokenizer.data.mel import MelConfig, MelSpectrogramFeature
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
        # "mel" (ECAPA), "waveform" (WavLM/ReDimNet), or "waveform_raw" (Qwen3)
        # — round-trips via model_cfg.encoder.
        self.input_kind = self.model.input_kind

        if mel_cfg is None:
            mel_cfg = MelConfig()  # defaults: 16 kHz / 128 mels / linear — matches BiCodec
        self.mel_cfg = mel_cfg

        # Determine the target sample rate for audio loading:
        # - For waveform backbones with a native sample_rate (e.g. Qwen3 @ 24kHz),
        #   use that rate so the backbone doesn't need to resample internally.
        # - Otherwise, use mel_cfg.sample_rate (16kHz).
        backbone_sr = getattr(self.model.speaker_encoder, "sample_rate", None)
        self._wav_sample_rate = backbone_sr if backbone_sr is not None else mel_cfg.sample_rate

        # Mel transform is only needed for the ECAPA (mel) path; for waveform
        # backbones the backbone consumes raw waveform and normalizes internally.
        self.mel = (
            MelSpectrogramFeature(mel_cfg).to(self.device)
            if self.input_kind == "mel"
            else None
        )

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _load_wav(self, path: Union[str, Path]) -> torch.Tensor:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)  # mono
        if sr != self._wav_sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=self._wav_sample_rate)
        return torch.from_numpy(wav).unsqueeze(0).to(self.device)  # (1, T)

    def _to_input(self, audio: AudioInput):
        """Prepare the model input + optional attention_mask.

        Returns:
            (inp, mask) where:
            - ECAPA:  inp = mel (B, n_mels, T),  mask = None
            - WavLM/ReDimNet: inp = waveform (B, T),  mask = None
            - Qwen3:  inp = log-mel (B, T, 128),  mask = (B, T) from processor
        """
        if isinstance(audio, (str, Path)):
            wav = self._load_wav(audio)  # (1, T)
        else:
            wav = audio.to(self.device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

        if self.input_kind == "waveform_raw":
            # Qwen3 path: use backbone's processor to compute mel + mask.
            backbone = self.model.speaker_encoder
            features = backbone.preprocess(wav.cpu(), sampling_rate=self._wav_sample_rate)
            mel = features["input_values"].to(self.device)
            mask = features["attention_mask"].to(self.device)
            return mel, mask

        if self.input_kind == "waveform":
            return wav, None  # (B, T)
        return self.mel(wav), None  # (B, n_mels, T)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tokenize(self, audio: AudioInput) -> torch.Tensor:
        inp, mask = self._to_input(audio)
        return self.model.tokenize(inp, attention_mask=mask)

    @torch.no_grad()
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.model.detokenize(indices.to(self.device))

    @torch.no_grad()
    def encode(self, audio: AudioInput) -> dict:
        """Return all forward outputs: x_vector, d_vector, indices."""
        inp, mask = self._to_input(audio)
        x_vec, d_vec, indices = self.model(inp, attention_mask=mask)
        return {"x_vector": x_vec, "d_vector": d_vec, "indices": indices}

    @torch.no_grad()
    def tokenize_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Tokenize from a precomputed mel (ECAPA path only)."""
        if self.input_kind != "mel":
            raise RuntimeError(
                "tokenize_from_mel is only valid for the ECAPA (mel) encoder; "
                f"this checkpoint uses input_kind={self.input_kind!r}."
            )
        return self.model.tokenize(mel.to(self.device))
