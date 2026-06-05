"""Batch-build LLM-TTS training data (JSONL of ``{"text": <sequence>}``).

For every clip in an LJSpeech-style CSV:
  1. speaker-tokenize the 16 kHz audio (trained speaker tokenizer) -> 32 ids,
  2. SNAC-tokenize the 24 kHz audio -> 7*frames audio tokens,
  3. assemble the ``<|task_tts|>...<|end_audio_token|>`` sequence,
  4. append one JSON line ``{"text": "<sequence>"}``.

Usage:
    PYTHONPATH=. python scripts/llm_tts/build_training_data.py --config-name base \
        data.csv_path=/path/metadata.csv data.wavs_dir=/path/wavs \
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
        output=./data/llm_tts_train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dragon_tts.llm_tts.data.csv_dataset import (
    CsvDatasetConfig,
    LJSpeechCsvDataset,
    collate_keep_lists,
)
from dragon_tts.llm_tts.sequence import build_sequence
from dragon_tts.llm_tts.snac_codec import SnacCodec
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline


def _fix_length(wav: torch.Tensor, target: int) -> torch.Tensor:
    """Center-crop (if longer) or repeat-pad (if shorter) to ``target`` samples.

    Matches the speaker tokenizer's training distribution (fixed-length crop)
    and lets us batch clips of different lengths together.
    """
    n = wav.shape[0]
    if n == target:
        return wav
    if n > target:
        start = (n - target) // 2
        return wav[start : start + target]
    reps = -(-target // max(1, n))  # ceil
    return wav.repeat(reps)[:target]


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "llm_tts"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    assert cfg.get("speaker_ckpt"), "Set speaker_ckpt=<path to .ckpt>"
    assert cfg.get("output"), "Set output=<path to .jsonl>"

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    ds_cfg = CsvDatasetConfig(
        csv_path=str(cfg.data.csv_path),
        wavs_dir=cfg.data.get("wavs_dir"),
        wav_ext=cfg.data.get("wav_ext", ".wav"),
        delimiter=cfg.data.get("delimiter", "|"),
        has_header=cfg.data.get("has_header", False),
        id_col=cfg.data.get("id_col", 0),
        text_col=cfg.data.get("text_col", 2),
        text_fallback_col=cfg.data.get("text_fallback_col", 1),
        speaker_col=cfg.data.get("speaker_col", None),
    )
    dataset = LJSpeechCsvDataset(ds_cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_keep_lists,
    )

    spk_pipe = SpeakerTokenizerPipeline(str(cfg.speaker_ckpt), device=device)
    snac = SnacCodec(device=device, model_id=cfg.get("snac_model"))

    speaker_crop = int(cfg.get("speaker_crop_seconds", 4.0) * ds_cfg.speaker_sr)
    skip_empty = cfg.get("skip_empty_text", True)

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(out_path.suffix + ".done")

    # Resume support: skip wav_paths already recorded in the sidecar.
    done: set[str] = set()
    resume = cfg.get("resume", False)
    if resume and done_path.exists():
        done = {ln.strip() for ln in done_path.read_text().splitlines() if ln.strip()}

    limit = cfg.get("limit", None)
    n_written = 0
    mode = "a" if resume else "w"
    with open(out_path, mode, encoding="utf-8") as out_f, open(
        done_path, "a" if resume else "w", encoding="utf-8"
    ) as done_f:
        for batch in tqdm(loader, desc="build"):
            # --- speaker tokens (batched over a fixed-length crop) ---
            wavs16 = [_fix_length(w, speaker_crop) for w in batch["wav_16k"]]
            wavs16 = torch.stack(wavs16, dim=0)  # (B, L)
            spk_idx = spk_pipe.tokenize(wavs16)  # (B, 1, 32)
            spk_idx = spk_idx[:, 0, :].cpu().tolist()  # B lists of 32 ids

            # --- SNAC audio tokens (batched) ---
            all_audio_tokens = snac.batch_encode(batch["wav_24k"])

            for i in range(len(batch["wav_path"])):
                wav_path = batch["wav_path"][i]
                if wav_path in done:
                    continue
                text = batch["text"][i]
                if skip_empty and not text:
                    continue
                seq = build_sequence(spk_idx[i], text, all_audio_tokens[i])
                out_f.write(json.dumps({"text": seq}, ensure_ascii=False) + "\n")
                done_f.write(wav_path + "\n")
                n_written += 1
                if limit and n_written >= limit:
                    out_f.flush()
                    print(f"[done] wrote {n_written} examples to {out_path} (limit hit)")
                    return

    print(f"[done] wrote {n_written} examples to {out_path}")


if __name__ == "__main__":
    main()
