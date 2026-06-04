"""LightningDataModule cho SpeakerAudioDataset."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dragon_tts.speaker_tokenizer.data.audio_dataset import (
    SpeakerAudioDataset,
    SpeakerAudioDatasetConfig,
    collate_mels,
    load_manifest,
    split_by_speaker,
)
from dragon_tts.speaker_tokenizer.data.mel import MelConfig


class SpeakerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest: str,
        mel_cfg: MelConfig,
        crop_seconds: float = 4.0,
        min_seconds: float = 1.0,
        val_speaker_frac: float = 0.05,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.manifest = manifest
        self.mel_cfg = mel_cfg
        self.crop_seconds = crop_seconds
        self.min_seconds = min_seconds
        self.val_speaker_frac = val_speaker_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_ds: Optional[SpeakerAudioDataset] = None
        self.val_ds: Optional[SpeakerAudioDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        items = load_manifest(self.manifest)
        train_items, val_items = split_by_speaker(
            items, self.val_speaker_frac, seed=self.seed
        )

        train_cfg = SpeakerAudioDatasetConfig(
            mel=self.mel_cfg,
            crop_seconds=self.crop_seconds,
            min_seconds=self.min_seconds,
            deterministic_crop=False,
        )
        val_cfg = SpeakerAudioDatasetConfig(
            mel=self.mel_cfg,
            crop_seconds=self.crop_seconds,
            min_seconds=self.min_seconds,
            deterministic_crop=True,
        )
        self.train_ds = SpeakerAudioDataset(train_items, train_cfg)
        self.val_ds = SpeakerAudioDataset(val_items, val_cfg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_mels,
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
            collate_fn=collate_mels,
            persistent_workers=self.num_workers > 0,
        )
