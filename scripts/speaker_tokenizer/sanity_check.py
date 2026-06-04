"""Sanity check: round-trip cosine/MSE, codebook utilization, entropy.

Usage:
    python scripts/sanity_check.py ckpt=<path> manifest=<eval.jsonl>
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dragon_tts.speaker_tokenizer.data.audio_dataset import (
    SpeakerAudioDataset,
    SpeakerAudioDatasetConfig,
    collate_inputs,
    load_manifest,
)
from dragon_tts.speaker_tokenizer.data.mel import MelConfig
from dragon_tts.speaker_tokenizer.training.lit_module import SpeakerTokenizerLit


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "speaker_tokenizer"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    assert "ckpt" in cfg, "Set ckpt=<path> on the CLI"
    assert "manifest" in cfg, "Set manifest=<eval.jsonl> on the CLI"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lit = SpeakerTokenizerLit.load_from_checkpoint(str(cfg.ckpt), map_location=device)
    lit.eval()
    lit.to(device)
    model = lit.model

    # "mel" (ECAPA) hoặc "waveform" (WavLM) — lấy từ encoder của checkpoint.
    input_kind = model.input_kind

    mel_cfg = MelConfig(**OmegaConf.to_container(cfg.mel, resolve=True))
    items = load_manifest(str(cfg.manifest))
    ds = SpeakerAudioDataset(
        items,
        SpeakerAudioDatasetConfig(
            mel=mel_cfg,
            crop_seconds=cfg.data.crop_seconds,
            min_seconds=cfg.data.min_seconds,
            deterministic_crop=True,
            input_kind=input_kind,
        ),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_inputs,
    )

    codebook_size = model.codebook_size
    token_num = model.token_num
    num_quant = model.fsq_num_quantizers

    # Per-position index counter for utilization & entropy.
    counts: Dict[tuple[int, int], Dict[int, int]] = {
        (t, q): {} for t in range(token_num) for q in range(num_quant)
    }
    cos_sims = []
    mses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            inp = batch["input"].to(device)
            x_vec, d_vec, indices = model(inp)
            cos_sims.append(F.cosine_similarity(d_vec, x_vec, dim=-1).cpu())
            mses.append(F.mse_loss(d_vec, x_vec, reduction="none").mean(dim=-1).cpu())

            # indices shape: (B, num_quantizers, token_num) — channels-first convention.
            idx_cpu = indices.cpu()
            for q in range(num_quant):
                for t in range(token_num):
                    vals = idx_cpu[:, q, t].tolist()
                    bucket = counts[(t, q)]
                    for v in vals:
                        bucket[v] = bucket.get(v, 0) + 1

    cos_sims = torch.cat(cos_sims)
    mses = torch.cat(mses)

    print(f"\n=== Round-trip metrics ({len(cos_sims)} utterances) ===")
    print(f"cosine(x_vec, d_vec): mean={cos_sims.mean():.4f}  median={cos_sims.median():.4f}  min={cos_sims.min():.4f}")
    print(f"MSE(x_vec, d_vec):    mean={mses.mean():.6f}  median={mses.median():.6f}  max={mses.max():.6f}")

    print(f"\n=== Codebook utilization (per token position) ===")
    print(f"codebook_size = {codebook_size}, token_num = {token_num}, num_quantizers = {num_quant}")
    flagged = 0
    entropies = []
    for q in range(num_quant):
        for t in range(token_num):
            bucket = counts[(t, q)]
            total = sum(bucket.values())
            util = len(bucket) / codebook_size
            if total == 0:
                continue
            ent = 0.0
            for c in bucket.values():
                p = c / total
                ent -= p * math.log2(p)
            entropies.append(ent)
            if util < 0.05:
                flagged += 1
                print(f"  [q={q} t={t:02d}] util={util:.3%} (only {len(bucket)} unique)  entropy={ent:.2f} bits")

    if not flagged:
        print("  ✔ all positions ≥ 5% utilization")
    util_per_pos = [len(b) / codebook_size for b in counts.values() if sum(b.values()) > 0]
    print(f"\nmean utilization = {sum(util_per_pos)/max(1,len(util_per_pos)):.3%}")
    print(f"mean entropy     = {sum(entropies)/max(1,len(entropies)):.2f} bits  (healthy ≈ {math.log2(codebook_size):.2f})")


if __name__ == "__main__":
    main()
