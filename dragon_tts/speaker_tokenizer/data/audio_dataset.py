"""Audio dataset đọc manifest JSONL và sinh mel-spectrogram."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torchaudio
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
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


@dataclass
class SpeakerAudioDatasetConfig:
    mel: MelConfig
    crop_seconds: float = 4.0
    min_seconds: float = 1.0
    deterministic_crop: bool = False  # True for val/eval
    # "mel"  → emit linear mel (B, n_mels, T) for the ECAPA backbone.
    # "waveform" → emit raw 16 kHz waveform (B, T) for the WavLM backbone.
    input_kind: str = "mel"


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
        self.crop_samples = int(cfg.crop_seconds * cfg.mel.sample_rate)
        self.min_samples = int(cfg.min_seconds * cfg.mel.sample_rate)

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[0]
        if n >= self.crop_samples:
            if self.cfg.deterministic_crop:
                start = max(0, (n - self.crop_samples) // 2)
            else:
                start = random.randint(0, n - self.crop_samples)
            return wav[start : start + self.crop_samples]
        # pad to crop_samples
        pad = self.crop_samples - n
        return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        wav = _load_wav(item["wav_path"], self.cfg.mel.sample_rate)
        if wav.shape[0] < self.min_samples:
            pad = self.min_samples - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))
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
