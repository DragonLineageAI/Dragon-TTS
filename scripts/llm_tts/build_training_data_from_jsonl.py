"""Batch-build LLM-TTS training data from pre-computed codec codes (JSONL).

Similar to ``build_training_data.py`` but reads *pre-computed* audio codes from
a JSONL file instead of encoding audio on-the-fly.  This avoids the expensive
codec forward pass when codes have already been extracted.

Input JSONL format (one JSON object per line)::

    {"audio_name": "EN_cFph1QMNggY_W000220", "text": "...", "codes": [2151, 43235, ...]}

- ``audio_name``: stem of the audio file (resolved inside ``audio_dir``).
- ``text``:       transcription.
- ``codes``:      pre-computed codec token *values* (flat list of ints).  For
                  NeuCodec this is 1 token per frame with values 0..65535; for
                  SNAC it would be the interleaved 7-token-per-frame values.

The script still needs audio files for speaker tokenization (the speaker
tokenizer runs on raw waveforms), but does **not** load a codec model.

Usage::

    PYTHONPATH=. python scripts/llm_tts/build_training_data_from_jsonl.py \\
        --config-name base \\
        jsonl_input=/path/to/precomputed.jsonl \\
        audio_dir=/path/to/wavs \\
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \\
        output=./data/llm_tts_train.jsonl

    # Specify codec type and audio extension:
    PYTHONPATH=. python scripts/llm_tts/build_training_data_from_jsonl.py \\
        --config-name base \\
        jsonl_input=/path/to/precomputed.jsonl \\
        audio_dir=/path/to/wavs \\
        audio_ext=.flac \\
        codec=neucodec \\
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \\
        output=./data/llm_tts_train.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

import hydra
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from dragon_tts.llm_tts.sequence import build_sequence
from dragon_tts.llm_tts.vocab import (
    NEUCODEC_CODEBOOK_SIZE,
    NEUCODEC_SLOTS,
    SNAC_CODEBOOK_SIZE,
    SNAC_SLOTS,
    audio_token,
)
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

SPEAKER_SR = 16000


# ── helpers ──────────────────────────────────────────────────────────────────


def _codec_params(codec_name: str) -> tuple[int, int]:
    """Return ``(slots_per_frame, codebook_size)`` for the named codec."""
    if codec_name == "neucodec":
        return NEUCODEC_SLOTS, NEUCODEC_CODEBOOK_SIZE
    return SNAC_SLOTS, SNAC_CODEBOOK_SIZE


def _codes_to_tokens(
    codes: List[int],
    slots_per_frame: int,
    codebook_size: int,
) -> List[str]:
    """Convert pre-computed code values into ``<|audio_*|>`` token strings.

    For NeuCodec (1 slot / frame): each value maps to ``audio_token(0, v)``.
    For SNAC (7 slots / frame): values are interleaved across 7 slots, cycling
    ``slot = i % 7``.
    """
    tokens: List[str] = []
    for i, v in enumerate(codes):
        slot = i % slots_per_frame
        tokens.append(audio_token(slot, v, codebook_size=codebook_size))
    return tokens


def _fix_length(wav: torch.Tensor, target: int) -> torch.Tensor:
    """Center-crop or repeat-pad ``wav`` to exactly ``target`` samples.

    Identical to the helper in ``build_training_data.py``.
    """
    n = wav.shape[0]
    if n == target:
        return wav
    if n > target:
        start = (n - target) // 2
        return wav[start : start + target]
    reps = -(-target // max(1, n))  # ceil
    return wav.repeat(reps)[:target]


def _count_lines(path: str) -> int:
    """Fast line count (binary, no JSON parsing) for tqdm total."""
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ── dataset ──────────────────────────────────────────────────────────────────


class PrecomputedJsonlDataset(IterableDataset):
    """Stream ``{audio_name, text, codes}`` from a JSONL line-by-line.

    **No upfront loading or indexing** — the file is read sequentially and
    each line is parsed on demand.  When ``num_workers > 0`` in the
    DataLoader, lines are sharded across workers via round-robin so that
    every line is processed exactly once.

    Each yielded item is a dict with:
      - ``wav_16k``: 16 kHz mono waveform for the speaker tokenizer.
      - ``text``: transcription string.
      - ``codes``: list of int (pre-computed codec codes).
      - ``audio_name``: identifier string.
    Items whose audio file is missing or unreadable are silently skipped.
    """

    def __init__(
        self,
        jsonl_path: str,
        audio_dir: str,
        audio_ext: str = ".wav",
    ):
        self.jsonl_path = jsonl_path
        self.audio_dir = Path(audio_dir)
        self.audio_ext = audio_ext

    def _resolve_audio_path(self, audio_name: str) -> Path:
        name = (
            audio_name
            if audio_name.endswith(self.audio_ext)
            else audio_name + self.audio_ext
        )
        return self.audio_dir / name

    def _process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        """Load audio and return the item dict, or ``None`` on failure."""
        audio_name = entry["audio_name"]
        text = entry.get("text", "")
        codes = entry["codes"]

        audio_path = self._resolve_audio_path(audio_name)

        if not audio_path.exists():
            log.warning("audio not found, skipping: %s", audio_path)
            return None

        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
        except Exception as exc:
            log.warning("failed to read %s: %s — skipping", audio_path, exc)
            return None

        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)

        wav_16k = (
            torchaudio.functional.resample(wav, sr, SPEAKER_SR)
            if sr != SPEAKER_SR
            else wav
        )

        return {
            "wav_16k": wav_16k,
            "text": text,
            "codes": codes,
            "audio_name": audio_name,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                # Round-robin sharding across DataLoader workers.
                if line_idx % num_workers != worker_id:
                    continue

                entry = json.loads(raw_line)
                item = self._process_entry(entry)
                if item is not None:
                    yield item


def _collate_keep_lists(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length items by keeping per-field lists (no stacking)."""
    if not batch:
        return {}
    return {k: [b[k] for b in batch] for k in batch[0]}


# ── main ─────────────────────────────────────────────────────────────────────


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "llm_tts"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    # -- validate required args ------------------------------------------------
    assert cfg.get("jsonl_input"), "Set jsonl_input=<path to .jsonl with pre-computed codes>"
    assert cfg.get("audio_dir"), "Set audio_dir=<directory containing audio files>"
    assert cfg.get("speaker_ckpt"), "Set speaker_ckpt=<path to .ckpt>"
    assert cfg.get("output"), "Set output=<path to .jsonl>"

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    # -- codec token mapping ---------------------------------------------------
    codec_name = cfg.get("codec", "snac")
    slots_per_frame, codebook_size = _codec_params(codec_name)

    # -- dataset & loader ------------------------------------------------------
    jsonl_path = str(cfg.jsonl_input)
    audio_ext = cfg.get("audio_ext", ".wav")
    dataset = PrecomputedJsonlDataset(
        jsonl_path=jsonl_path,
        audio_dir=str(cfg.audio_dir),
        audio_ext=audio_ext,
    )

    # Fast line count for the progress bar (binary scan, no JSON parsing).
    total_lines = _count_lines(jsonl_path)
    total_batches = -(-total_lines // cfg.data.batch_size)  # ceil

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate_keep_lists,
    )

    # -- speaker tokenizer -----------------------------------------------------
    spk_pipe = SpeakerTokenizerPipeline(str(cfg.speaker_ckpt), device=device)

    speaker_crop = int(cfg.get("speaker_crop_seconds", 4.0) * SPEAKER_SR)
    skip_empty = cfg.get("skip_empty_text", True)

    # -- output / resume -------------------------------------------------------
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(out_path.suffix + ".done")

    done: set[str] = set()
    resume = cfg.get("resume", False)
    if resume and done_path.exists():
        done = {ln.strip() for ln in done_path.read_text().splitlines() if ln.strip()}

    limit = cfg.get("limit", None)
    n_written = 0
    n_skipped = 0
    mode = "a" if resume else "w"

    with open(out_path, mode, encoding="utf-8") as out_f, open(
        done_path, "a" if resume else "w", encoding="utf-8"
    ) as done_f:
        for batch in tqdm(loader, desc="build (pre-computed codes)", total=total_batches):
            if not batch:
                continue

            # --- speaker tokens (batched over a fixed-length crop) ---
            wavs16 = [_fix_length(w, speaker_crop) for w in batch["wav_16k"]]
            wavs16 = torch.stack(wavs16, dim=0)  # (B, L)
            spk_idx = spk_pipe.tokenize(wavs16)   # (B, 1, 32)
            spk_idx = spk_idx[:, 0, :].cpu().tolist()  # B lists of 32 ids

            for i in range(len(batch["audio_name"])):
                audio_name = batch["audio_name"][i]
                if audio_name in done:
                    n_skipped += 1
                    continue

                text = batch["text"][i]
                if skip_empty and not text:
                    continue

                # Convert pre-computed codes → <|audio_*|> token strings.
                audio_tokens = _codes_to_tokens(
                    batch["codes"][i], slots_per_frame, codebook_size
                )

                seq = build_sequence(spk_idx[i], text, audio_tokens)
                out_f.write(json.dumps({"text": seq}, ensure_ascii=False) + "\n")
                done_f.write(audio_name + "\n")
                n_written += 1

                if limit and n_written >= limit:
                    out_f.flush()
                    print(
                        f"[done] wrote {n_written} examples to {out_path} (limit hit)"
                    )
                    return

    if n_skipped:
        print(f"[info] skipped {n_skipped} already-done entries")
    print(f"[done] wrote {n_written} examples to {out_path}")


if __name__ == "__main__":
    main()
