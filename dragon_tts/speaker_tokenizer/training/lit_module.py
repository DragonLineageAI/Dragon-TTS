"""PyTorch Lightning module cho SpeakerTokenizer.

Loss = mse_weight * MSE(d_vec, x_vec.detach()) + cosine_weight * (1 - cos(d_vec, x_vec)).mean()
Mặc định cosine_weight = 0 → pure MSE, đặt > 0 trong config để bật loss phụ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer


@dataclass
class LossWeights:
    mse: float = 1.0
    cosine: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 200000
    min_lr_ratio: float = 1e-2  # min_lr = lr * min_lr_ratio


def _cosine_warmup_lr(step: int, optim_cfg: OptimConfig) -> float:
    if step < optim_cfg.warmup_steps:
        return float(step) / max(1, optim_cfg.warmup_steps)
    progress = (step - optim_cfg.warmup_steps) / max(
        1, optim_cfg.max_steps - optim_cfg.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim_cfg.min_lr_ratio + (1.0 - optim_cfg.min_lr_ratio) * cosine


class SpeakerTokenizerLit(pl.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_weights: Dict[str, float],
        optim_cfg: Dict[str, Any],
        ecapa_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SpeakerTokenizer(**model_cfg)
        if ecapa_ckpt is not None:
            sd = torch.load(ecapa_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.model.load_ecapa_state_dict(sd, strict=True)
        else:
            # Vẫn freeze (random ECAPA = nonsense, nhưng đảm bảo invariants).
            self.model.freeze_ecapa()

        self.loss_weights = LossWeights(**loss_weights)
        self.optim_cfg = OptimConfig(**optim_cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mel: torch.Tensor):
        return self.model(mel)

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------

    def _compute_losses(
        self, x_vec: torch.Tensor, d_vec: torch.Tensor, indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        target = x_vec.detach()
        mse = F.mse_loss(d_vec, target)
        cos_sim = F.cosine_similarity(d_vec, target, dim=-1)  # (B,)
        cos_loss = (1.0 - cos_sim).mean()

        total = self.loss_weights.mse * mse + self.loss_weights.cosine * cos_loss

        # Codebook usage: indices shape (B, num_quantizers, token_num) — convention
        # channels-first khớp với SparkVox SpeakerEncoder.
        codebook_size = self.model.codebook_size
        b, nq, tn = indices.shape
        unique_counts = []
        for q in range(nq):
            for t in range(tn):
                unique_counts.append(indices[:, q, t].unique().numel())
        avg_unique = sum(unique_counts) / max(1, len(unique_counts))
        utilization = avg_unique / codebook_size

        return {
            "loss": total,
            "mse": mse.detach(),
            "cos_loss": cos_loss.detach(),
            "cos_sim_mean": cos_sim.mean().detach(),
            "codebook_util_batch": torch.tensor(utilization, device=mse.device),
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        mel = batch["mel"]
        x_vec, d_vec, indices = self.model(mel)
        losses = self._compute_losses(x_vec, d_vec, indices)

        bs = mel.shape[0]
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=(k in {"loss", "cos_sim_mean"}),
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        mel = batch["mel"]
        x_vec, d_vec, indices = self.model(mel)
        losses = self._compute_losses(x_vec, d_vec, indices)

        bs = mel.shape[0]
        for k, v in losses.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=(k in {"mse", "cos_sim_mean"}),
                batch_size=bs,
                sync_dist=True,
            )
        return losses["loss"]

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def _trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def configure_optimizers(self):
        params = list(self._trainable_parameters())
        opt = torch.optim.AdamW(
            params,
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: _cosine_warmup_lr(step, self.optim_cfg)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
