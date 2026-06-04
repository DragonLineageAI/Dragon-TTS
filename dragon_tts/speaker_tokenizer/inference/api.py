"""Public inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

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
        # "mel" (ECAPA) or "waveform" (WavLM) — round-trips via model_cfg.encoder.
        self.input_kind = self.model.input_kind

        if mel_cfg is None:
            mel_cfg = MelConfig()  # defaults: 16 kHz / 128 mels / linear — matches BiCodec
        self.mel_cfg = mel_cfg
        # Mel transform is only needed for the ECAPA (mel) path; for WavLM the
        # backbone consumes raw waveform and normalizes internally.
        self.mel = (
            MelSpectrogramFeature(mel_cfg).to(self.device)
            if self.input_kind == "mel"
            else None
        )

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

    def _to_input(self, audio: AudioInput) -> torch.Tensor:
        """Prepare the model input: mel (B, n_mels, T) for ECAPA, or raw
        waveform (B, T) for WavLM, depending on the loaded encoder."""
        if isinstance(audio, (str, Path)):
            wav = self._load_wav(audio)  # (1, T)
        else:
            wav = audio.to(self.device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        if self.input_kind == "waveform":
            return wav  # (B, T)
        return self.mel(wav)  # (B, n_mels, T)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tokenize(self, audio: AudioInput) -> torch.Tensor:
        inp = self._to_input(audio)
        return self.model.tokenize(inp)

    @torch.no_grad()
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.model.detokenize(indices.to(self.device))

    @torch.no_grad()
    def encode(self, audio: AudioInput) -> dict:
        """Return all forward outputs: x_vector, d_vector, indices."""
        inp = self._to_input(audio)
        x_vec, d_vec, indices = self.model(inp)
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
