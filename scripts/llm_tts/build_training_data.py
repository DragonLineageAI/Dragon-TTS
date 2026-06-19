"""Batch-build LLM-TTS training data (JSONL of ``{"text": <sequence>}``).

For every clip in an LJSpeech-style CSV:
  1. speaker-tokenize the audio (trained speaker tokenizer) -> 32 ids,
  2. audio-tokenize with the selected codec (SNAC or NeuCodec),
  3. assemble the ``<|task_tts|>...<|end_audio_token|>`` sequence,
  4. append one JSON line ``{"text": "<sequence>"}``.

Usage:
    PYTHONPATH=. python scripts/llm_tts/build_training_data.py --config-name base \
        data.csv_path=/path/metadata.csv data.wavs_dir=/path/wavs \
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
        output=./data/llm_tts_train.jsonl

    # Use NeuCodec instead of SNAC:
    PYTHONPATH=. python scripts/llm_tts/build_training_data.py --config-name base \
        codec=neucodec \
        data.csv_path=/path/metadata.csv data.wavs_dir=/path/wavs \
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
        output=./data/llm_tts_train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.data.csv_dataset import (
    CsvDatasetConfig,
    LJSpeechCsvDataset,
    collate_keep_lists,
)
from dragon_tts.llm_tts.sequence import build_sequence
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline


def _build_codec(cfg: DictConfig, device: str) -> AudioCodec:
    """Instantiate the configured audio codec."""
    codec_name = cfg.get("codec", "snac")
    if codec_name == "neucodec":
        from dragon_tts.llm_tts.neucodec_codec import NeucodecCodec

        return NeucodecCodec(
            device=device, model_id=cfg.get("neucodec_model", "neuphonic/neucodec")
        )
    else:
        from dragon_tts.llm_tts.snac_codec import SnacCodec

        return SnacCodec(
            device=device,
            model_id=cfg.get("snac_model", "hubertsiuzdak/snac_24khz"),
        )




@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "llm_tts"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    assert cfg.get("speaker_ckpt"), "Set speaker_ckpt=<path to .ckpt>"
    assert cfg.get("output"), "Set output=<path to .jsonl>"

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    codec = _build_codec(cfg, device)
    codec_sr = codec.sample_rate

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
        codec_sr=codec_sr,
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
    spk_sr = spk_pipe._wav_sample_rate  # e.g. 24000 for Qwen3, 16000 for ECAPA

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
            # --- speaker tokens (full-length, resampled to backbone SR) ---
            wavs_resampled = []
            for w, sr in zip(batch["wav"], batch["sr"]):
                if sr != spk_sr:
                    w = torchaudio.functional.resample(w, sr, spk_sr)
                wavs_resampled.append(w)

            # Process each utterance individually (variable lengths).
            spk_idx = []
            for w in wavs_resampled:
                idx = spk_pipe.tokenize(w.unsqueeze(0))  # (1, 1, 32)
                spk_idx.append(idx[0, 0, :].cpu().tolist())  # list of 32 ids

            # --- audio tokens (batched via codec) ---
            all_audio_tokens = codec.batch_encode(batch["wav_codec"])

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
