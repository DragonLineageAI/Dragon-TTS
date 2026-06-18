"""LightningDataModule cho SpeakerAudioDataset."""

from __future__ import annotations

from typing import Iterable, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dragon_tts.speaker_tokenizer.data.audio_dataset import (
    Qwen3Collator,
    SpeakerAudioDataset,
    SpeakerAudioDatasetConfig,
    collate_inputs,
    load_manifest,
    split_by_speaker,
)
from dragon_tts.speaker_tokenizer.data.mel import MelConfig


class SpeakerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest: Union[str, Iterable[str]],
        mel_cfg: MelConfig,
        val_manifest: Optional[Union[str, Iterable[str]]] = None,
        crop_seconds: float = 4.0,
        min_seconds: float = 1.0,
        val_speaker_frac: float = 0.05,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0,
        input_kind: str = "mel",
        pad_mode: str = "repeat",
        target_sample_rate: Optional[int] = None,
        # Maximum audio duration (seconds) for waveform_raw mode.
        # Audio exceeding this is truncated.  Prevents 32-bit index overflow.
        max_seconds: Optional[float] = 30.0,
        # Qwen3 only: path to pretrained checkpoint for the processor collator.
        # When set (and input_kind=="waveform_raw"), the DataModule uses
        # ``Qwen3Collator`` which delegates mel + padding + masking to
        # ``EcapaTdnnFeatureExtractor``.
        qwen3_pretrained: Optional[str] = None,
    ):
        super().__init__()
        self.manifest = manifest
        self.val_manifest = val_manifest
        self.mel_cfg = mel_cfg
        self.crop_seconds = crop_seconds
        self.min_seconds = min_seconds
        self.val_speaker_frac = val_speaker_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.input_kind = input_kind
        self.pad_mode = pad_mode
        self.target_sample_rate = target_sample_rate
        self.max_seconds = max_seconds
        self.qwen3_pretrained = qwen3_pretrained

        self.train_ds: Optional[SpeakerAudioDataset] = None
        self.val_ds: Optional[SpeakerAudioDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.val_manifest is not None:
            # Val là manifest riêng → dùng trọn vẹn train + val, không split.
            train_items = load_manifest(self.manifest)
            val_items = load_manifest(self.val_manifest)
        else:
            # Không có val manifest → tách held-out theo speaker từ train.
            items = load_manifest(self.manifest)
            train_items, val_items = split_by_speaker(
                items, self.val_speaker_frac, seed=self.seed
            )

        train_cfg = SpeakerAudioDatasetConfig(
            mel=self.mel_cfg,
            crop_seconds=self.crop_seconds,
            min_seconds=self.min_seconds,
            deterministic_crop=False,
            input_kind=self.input_kind,
            pad_mode=self.pad_mode,
            target_sample_rate=self.target_sample_rate,
            max_seconds=self.max_seconds,
        )
        val_cfg = SpeakerAudioDatasetConfig(
            mel=self.mel_cfg,
            crop_seconds=self.crop_seconds,
            min_seconds=self.min_seconds,
            deterministic_crop=True,
            input_kind=self.input_kind,
            pad_mode=self.pad_mode,
            target_sample_rate=self.target_sample_rate,
            max_seconds=self.max_seconds,
        )
        self.train_ds = SpeakerAudioDataset(train_items, train_cfg)
        self.val_ds = SpeakerAudioDataset(val_items, val_cfg)

    def _collate_fn(self):
        if self.input_kind == "waveform_raw" and self.qwen3_pretrained:
            return Qwen3Collator(
                self.qwen3_pretrained,
                self.target_sample_rate or 24000,
            )
        return collate_inputs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn(),
            persistent_workers=self.num_workers > 0,
        )
