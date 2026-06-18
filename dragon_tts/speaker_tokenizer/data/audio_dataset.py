"""Audio dataset đọc manifest JSONL và sinh mel-spectrogram."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from dragon_tts.speaker_tokenizer.data.mel import MelConfig, MelSpectrogramFeature


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    """Load wav, downmix to mono, resample to target_sr. Returns (T,)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    # (samples, channels) → mono
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(wav)


@dataclass
class SpeakerAudioDatasetConfig:
    mel: MelConfig
    crop_seconds: float = 4.0
    min_seconds: float = 1.0
    deterministic_crop: bool = False  # True for val/eval
    # "mel"          → emit linear mel (B, n_mels, T) for the ECAPA backbone.
    # "waveform"     → emit raw waveform (B, T) for WavLM / ReDimNet backbones
    #                   (fixed-length via crop/extend).
    # "waveform_raw" → emit raw waveform WITHOUT crop/extend (variable length),
    #                   for backbones that handle masking natively (e.g. Qwen3).
    #                   Requires ``collate_variable_length`` as collate_fn.
    input_kind: str = "mel"
    # Cách kéo dài clip ngắn hơn crop_seconds:
    #   "zero"   → đệm 0 (silence) ở cuối.
    #   "repeat" → lặp (tile) waveform cho đến đủ độ dài → cửa sổ toàn giọng thật,
    #              tránh contaminate pooling/normalize bằng silence.
    pad_mode: str = "repeat"
    # Target sample rate for waveform output. If None, defaults to mel.sample_rate.
    # Set to backbone's native rate (e.g. 24000 for Qwen3) so the dataset emits
    # waveform at the correct rate without backbone-internal resampling.
    target_sample_rate: Optional[int] = None


class SpeakerAudioDataset(Dataset):
    """Mỗi item: `{wav_path, speaker_id?}` → `input` (mel hoặc waveform).

    `input` là mel `(n_mels, T_frames)` khi `input_kind=="mel"`, hoặc waveform
    `(T_samples,)` khi `input_kind=="waveform"`.
    """

    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        cfg: SpeakerAudioDatasetConfig,
    ):
        self.items = manifest
        self.cfg = cfg
        self.mel_fn = MelSpectrogramFeature(cfg.mel)
        # Use target_sample_rate if set, otherwise fall back to mel config.
        self._wav_sr = cfg.target_sample_rate or cfg.mel.sample_rate
        self.crop_samples = int(cfg.crop_seconds * self._wav_sr)
        self.min_samples = int(cfg.min_seconds * self._wav_sr)

    def __len__(self) -> int:
        return len(self.items)

    def _extend(self, wav: torch.Tensor, target: int) -> torch.Tensor:
        """Kéo dài wav lên `target` mẫu theo `pad_mode` (zero hoặc repeat)."""
        n = wav.shape[0]
        if n >= target:
            return wav
        if n > 0 and self.cfg.pad_mode == "repeat":
            reps = -(-target // n)  # ceil(target / n)
            return wav.repeat(reps)[:target]
        return torch.nn.functional.pad(wav, (0, target - n))

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[0]
        if n >= self.crop_samples:
            if self.cfg.deterministic_crop:
                start = max(0, (n - self.crop_samples) // 2)
            else:
                start = random.randint(0, n - self.crop_samples)
            return wav[start : start + self.crop_samples]
        return self._extend(wav, self.crop_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        wav = _load_wav(item["wav_path"], self._wav_sr)

        if self.cfg.input_kind == "waveform_raw":
            # Qwen3 path: full-length waveform, no crop/extend.
            inp = wav  # (T_samples,) — variable length
        else:
            if wav.shape[0] < self.min_samples:
                wav = self._extend(wav, self.min_samples)
            wav = self._crop(wav)
            if self.cfg.input_kind == "waveform":
                inp = wav  # (T_samples,)
            else:
                inp = self.mel_fn(wav.unsqueeze(0)).squeeze(0)  # (n_mels, T)

        return {
            "input": inp,
            "speaker_id": item.get("speaker_id", ""),
            "wav_path": item["wav_path"],
        }


def collate_inputs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack the per-item `input` tensor (mel or waveform; uniform length)."""
    inputs = torch.stack([b["input"] for b in batch], dim=0)
    return {
        "input": inputs,
        "speaker_id": [b["speaker_id"] for b in batch],
        "wav_path": [b["wav_path"] for b in batch],
    }


class Qwen3Collator:
    """Collate for Qwen3: delegates to ``EcapaTdnnFeatureExtractor``.

    The processor handles mel computation (librosa + torch.stft), resampling,
    padding to max length, and ``attention_mask`` generation — all in one call.

    Used with ``input_kind="waveform_raw"`` where the dataset emits full-length
    raw waveform tensors (no crop/extend).

    Returns:
        input: (B, T_mel, n_mels) log-mel spectrogram, padded.
        attention_mask: (B, T_mel) long — 1 for real frames, 0 for padding.
    """

    def __init__(self, pretrained: str, sample_rate: int = 24000):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self.sample_rate = sample_rate

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Convert waveform tensors to numpy for the processor.
        wavs = [b["input"].numpy() for b in batch]
        features = self.processor(wavs, sampling_rate=self.sample_rate)
        return {
            "input": features["input_values"],            # (B, T_mel, n_mels)
            "attention_mask": features["attention_mask"],  # (B, T_mel)
            "speaker_id": [b["speaker_id"] for b in batch],
            "wav_path": [b["wav_path"] for b in batch],
        }


# Backward-compatible alias.
collate_mels = collate_inputs


def load_manifest(
    path: Union[str, Iterable[str]]
) -> List[Dict[str, Any]]:
    """Đọc một hoặc nhiều file JSONL. `path` có thể là str hoặc list[str]
    (kể cả Hydra ListConfig). Nhiều file sẽ được nối lại theo thứ tự."""
    if isinstance(path, str):
        return _load_manifest(path)
    items: List[Dict[str, Any]] = []
    for p in path:
        items.extend(_load_manifest(p))
    return items


def split_by_speaker(
    items: List[Dict[str, Any]], val_frac: float, seed: int = 0
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chia train/val theo speaker_id. Nếu không có speaker_id thì fallback theo wav."""
    have_speaker = all("speaker_id" in it for it in items)
    if not have_speaker:
        rng = random.Random(seed)
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        return shuffled[n_val:], shuffled[:n_val]

    speakers = sorted({it["speaker_id"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * val_frac))
    val_spk = set(speakers[:n_val_spk])
    train = [it for it in items if it["speaker_id"] not in val_spk]
    val = [it for it in items if it["speaker_id"] in val_spk]
    return train, val
