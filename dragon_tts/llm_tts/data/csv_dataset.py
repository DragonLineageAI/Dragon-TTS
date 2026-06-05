"""LJSpeech-style CSV dataset for the LLM-TTS data builder.

Reads a metadata CSV (default LJSpeech: ``id|transcription|normalized_transcription``,
pipe-delimited, no header, wavs under a ``wavs/`` dir) and yields, per clip, the
audio resampled to *both* sample rates needed downstream:

  - 24 kHz for SNAC audio tokenization,
  - 16 kHz for the speaker tokenizer.

Audio is loaded once at native sample rate and resampled twice (no cropping —
SNAC needs the full clip).
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

SPEAKER_SR = 16000
SNAC_SR = 24000


@dataclass
class CsvDatasetConfig:
    csv_path: str
    # If set, wav path = wavs_dir/{id_col value}{wav_ext}; else id_col holds a path.
    wavs_dir: Optional[str] = None
    wav_ext: str = ".wav"
    delimiter: str = "|"
    has_header: bool = False
    id_col: int = 0
    # Default to LJSpeech normalized transcription (col 2), fall back to col 1.
    text_col: int = 2
    text_fallback_col: Optional[int] = 1
    speaker_col: Optional[int] = None
    speaker_sr: int = SPEAKER_SR
    snac_sr: int = SNAC_SR


def _read_rows(cfg: CsvDatasetConfig) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(cfg.csv_path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f, delimiter=cfg.delimiter)
        for i, row in enumerate(reader):
            if i == 0 and cfg.has_header:
                continue
            if not row or all(c.strip() == "" for c in row):
                continue
            rows.append(row)
    return rows


def _resolve_wav(cfg: CsvDatasetConfig, row: List[str]) -> str:
    raw = row[cfg.id_col].strip()
    if cfg.wavs_dir is None:
        return raw
    name = raw if raw.endswith(cfg.wav_ext) else raw + cfg.wav_ext
    return str(Path(cfg.wavs_dir) / name)


def _resolve_text(cfg: CsvDatasetConfig, row: List[str]) -> str:
    if cfg.text_col < len(row) and row[cfg.text_col].strip():
        return row[cfg.text_col].strip()
    if cfg.text_fallback_col is not None and cfg.text_fallback_col < len(row):
        return row[cfg.text_fallback_col].strip()
    return ""


class LJSpeechCsvDataset(Dataset):
    """Each item: ``{wav_24k, wav_16k, text, speaker_id, wav_path}``."""

    def __init__(self, cfg: CsvDatasetConfig):
        self.cfg = cfg
        self.rows = _read_rows(cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cfg = self.cfg
        row = self.rows[idx]
        wav_path = _resolve_wav(cfg, row)
        text = _resolve_text(cfg, row)
        speaker_id = (
            row[cfg.speaker_col].strip()
            if cfg.speaker_col is not None and cfg.speaker_col < len(row)
            else ""
        )

        wav, sr = torchaudio.load(wav_path)  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)

        wav_24k = (
            torchaudio.functional.resample(wav, sr, cfg.snac_sr)
            if sr != cfg.snac_sr
            else wav
        )
        wav_16k = (
            torchaudio.functional.resample(wav, sr, cfg.speaker_sr)
            if sr != cfg.speaker_sr
            else wav
        )
        return {
            "wav_24k": wav_24k,
            "wav_16k": wav_16k,
            "text": text,
            "speaker_id": speaker_id,
            "wav_path": wav_path,
        }


def collate_keep_lists(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length clips by keeping per-field lists (no stacking)."""
    return {k: [b[k] for b in batch] for k in batch[0]}
