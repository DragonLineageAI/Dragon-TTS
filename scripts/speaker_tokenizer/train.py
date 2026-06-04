"""Hydra entry point — train SpeakerTokenizer."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from dragon_tts.speaker_tokenizer.data.datamodule import SpeakerDataModule
from dragon_tts.speaker_tokenizer.data.mel import MelConfig
from dragon_tts.speaker_tokenizer.training.lit_module import SpeakerTokenizerLit


def _to_python(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "speaker_tokenizer"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    mel_cfg = MelConfig(**_to_python(cfg.mel))

    dm = SpeakerDataModule(
        manifest=cfg.data.manifest,
        val_manifest=cfg.data.get("val_manifest", None),
        mel_cfg=mel_cfg,
        crop_seconds=cfg.data.crop_seconds,
        min_seconds=cfg.data.min_seconds,
        val_speaker_frac=cfg.data.val_speaker_frac,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.data.seed,
    )

    lit = SpeakerTokenizerLit(
        model_cfg=_to_python(cfg.model),
        loss_weights=_to_python(cfg.loss_weights),
        optim_cfg=_to_python(cfg.optim),
        ecapa_ckpt=str(cfg.ecapa_ckpt) if cfg.ecapa_ckpt else None,
    )

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(out_dir / "ckpts"),
            filename="best-{epoch:03d}-{val/mse:.4f}",
            monitor="val/mse",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger(save_dir=str(out_dir), name="tb")

    trainer_kwargs = dict(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(out_dir),
    )
    if cfg.trainer.max_steps and int(cfg.trainer.max_steps) > 0:
        trainer_kwargs["max_steps"] = int(cfg.trainer.max_steps)
    if cfg.trainer.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.trainer.limit_train_batches
    if cfg.trainer.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = cfg.trainer.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
