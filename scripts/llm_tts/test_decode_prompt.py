#!/usr/bin/env python3
"""Decode audio tokens from an LLM-TTS prompt back to a WAV file.

Takes a full training sequence (or the generated output from the LLM) that
contains ``<|start_audio_token|>...<|end_audio_token|>`` and reconstructs
the audio waveform using the selected codec (SNAC or NeuCodec).

Usage examples:

    # From a direct prompt string (SNAC, default):
    PYTHONPATH=. python scripts/llm_tts/test_decode_prompt.py \
        --prompt '<|task_tts|><|start_embedding_token|>...<|end_audio_token|>' \
        --output decoded.wav

    # From a JSONL file with NeuCodec:
    PYTHONPATH=. python scripts/llm_tts/test_decode_prompt.py \
        --codec neucodec \
        --file data/llm_tts_train.jsonl \
        --line 0 \
        --output decoded.wav

    # From stdin (pipe a sequence in):
    echo '<|task_tts|>...<|end_audio_token|>' | \
        PYTHONPATH=. python scripts/llm_tts/test_decode_prompt.py --output decoded.wav

    # Use CPU instead of GPU:
    PYTHONPATH=. python scripts/llm_tts/test_decode_prompt.py \
        --prompt '...' --device cpu --output decoded.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.sequence import parse_sequence
from dragon_tts.llm_tts.vocab import (
    NEUCODEC_CODEBOOK_SIZE,
    NEUCODEC_SLOTS,
    SNAC_CODEBOOK_SIZE,
    SNAC_SLOTS,
)


def _build_codec(codec_name: str, device: str, model_id: str | None) -> AudioCodec:
    """Instantiate the configured audio codec."""
    if codec_name == "neucodec":
        from dragon_tts.llm_tts.neucodec_codec import NeucodecCodec

        kwargs = {"device": device}
        if model_id:
            kwargs["model_id"] = model_id
        return NeucodecCodec(**kwargs)
    else:
        from dragon_tts.llm_tts.snac_codec import SnacCodec

        kwargs = {"device": device}
        if model_id:
            kwargs["model_id"] = model_id
        return SnacCodec(**kwargs)


def read_sequence(args: argparse.Namespace) -> str:
    """Obtain the raw sequence string from --prompt, --file, or stdin."""
    if args.prompt:
        return args.prompt

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx == args.line:
                        obj = json.loads(line)
                        return obj.get("text", line.strip())
            raise IndexError(
                f"Line {args.line} not found in {path} "
                f"(file has only {idx + 1} lines)"
            )
        else:
            # Plain text file — read entire contents
            return path.read_text(encoding="utf-8").strip()

    # Fallback: read from stdin
    if sys.stdin.isatty():
        print("Paste the sequence and press Ctrl+D (EOF):", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode audio tokens from an LLM-TTS sequence to WAV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Raw sequence string containing audio tokens.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .jsonl or text file containing the sequence.",
    )
    parser.add_argument(
        "--line",
        type=int,
        default=0,
        help="Line index to read when --file is a .jsonl (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="decoded_output.wav",
        help="Output WAV file path (default: decoded_output.wav).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for codec model: cuda or cpu (default: cuda).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="snac",
        choices=["snac", "neucodec"],
        help="Audio codec to use for decoding (default: snac).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override codec model id (e.g. hubertsiuzdak/snac_24khz or neuphonic/neucodec).",
    )
    args = parser.parse_args()

    # ── Codec-specific constants ───────────────────────────────────────
    if args.codec == "neucodec":
        slots = NEUCODEC_SLOTS
        cb_size = NEUCODEC_CODEBOOK_SIZE
    else:
        slots = SNAC_SLOTS
        cb_size = SNAC_CODEBOOK_SIZE

    # ── 1. Read the sequence ───────────────────────────────────────────
    seq = read_sequence(args)
    if not seq:
        print("ERROR: empty sequence. Provide --prompt, --file, or pipe to stdin.",
              file=sys.stderr)
        sys.exit(1)

    # ── 2. Parse the sequence ──────────────────────────────────────────
    print("Parsing sequence...")
    parsed = parse_sequence(seq, codebook_size=cb_size)

    spk_ids = parsed["spk_ids"]
    text = parsed["text"]
    audio_pairs = parsed["audio_pairs"]

    n_frames = len(audio_pairs) // slots
    n_leftover = len(audio_pairs) % slots

    print(f"  Codec        : {args.codec} (slots={slots}, codebook={cb_size})")
    print(f"  Speaker IDs  : {len(spk_ids)} tokens → {spk_ids[:8]}{'...' if len(spk_ids) > 8 else ''}")
    print(f"  Text         : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Audio pairs  : {len(audio_pairs)} tokens → {n_frames} complete frames")
    if n_leftover:
        print(f"  ⚠ {n_leftover} leftover tokens (not a multiple of {slots}; will be dropped)")

    if n_frames == 0:
        print("ERROR: no complete audio frames found in the sequence.", file=sys.stderr)
        sys.exit(1)

    # ── 3. Decode with codec ───────────────────────────────────────────
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Loading {args.codec} model on '{device}'...")

    codec = _build_codec(args.codec, device, args.model_id)

    print("Decoding audio...")
    waveform = codec.decode_pairs(audio_pairs)  # (T,) 1-D tensor

    out_sr = codec.output_sample_rate
    duration = waveform.shape[0] / out_sr
    print(f"  Decoded waveform: {waveform.shape[0]} samples, {duration:.2f}s @ {out_sr} Hz")

    # ── 4. Save WAV ───────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform.numpy(), out_sr)
    print(f"✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
