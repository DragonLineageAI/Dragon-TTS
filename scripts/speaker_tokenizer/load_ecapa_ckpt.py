"""Trích state_dict ECAPA-TDNN từ checkpoint user cung cấp.

Hỗ trợ:
  - Checkpoint BiCodec đầy đủ: keys có dạng
      `model.generator.speaker_encoder.speaker_encoder.<X>` hoặc
      `speaker_encoder.speaker_encoder.<X>`
  - Checkpoint ECAPA standalone: keys đã ở đúng dạng `<X>`.

Output là dict trực tiếp khớp với `ECAPA_TDNN_GLOB_c512(feat_dim=128, embed_dim=1024)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from dragon_tts.modules.ecapa.ecapa_tdnn import ECAPA_TDNN_GLOB_c512


# Tiền tố ưu tiên (trái sang phải).
PREFIX_CANDIDATES = [
    "model.generator.speaker_encoder.speaker_encoder.",
    "generator.speaker_encoder.speaker_encoder.",
    "speaker_encoder.speaker_encoder.",
    "module.speaker_encoder.",
    "speaker_encoder.",
    "",
]


def _strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Tự động dò prefix khớp với ECAPA_TDNN_GLOB_c512."""
    reference = ECAPA_TDNN_GLOB_c512(feat_dim=128, embed_dim=1024).state_dict()
    ref_keys = set(reference.keys())

    for prefix in PREFIX_CANDIDATES:
        if prefix == "":
            keys_after = set(state_dict.keys())
        else:
            keys_after = {
                k[len(prefix):] for k in state_dict.keys() if k.startswith(prefix)
            }
        if not keys_after:
            continue
        overlap = keys_after & ref_keys
        if len(overlap) >= 0.9 * len(ref_keys):
            print(f"[load_ecapa_ckpt] matched prefix: {prefix!r} (overlap={len(overlap)}/{len(ref_keys)})")
            return {
                (k[len(prefix):] if prefix else k): v
                for k, v in state_dict.items()
                if (k.startswith(prefix) if prefix else True)
            }

    raise RuntimeError(
        f"Cannot find a matching prefix for ECAPA keys. Tried: {PREFIX_CANDIDATES}. "
        f"Inspect your checkpoint manually."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source checkpoint (.pt/.ckpt)")
    parser.add_argument("--dst", required=True, help="Destination ECAPA state_dict (.pt)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(str(src), device="cpu")
    else:
        raw = torch.load(src, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(raw)}")

    sd = _strip_prefix(raw)

    # Verify by loading into reference model.
    ref = ECAPA_TDNN_GLOB_c512(feat_dim=128, embed_dim=1024)
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    print(f"[load_ecapa_ckpt] missing keys: {len(missing)}")
    for k in missing:
        print(f"  - missing: {k}")
    print(f"[load_ecapa_ckpt] unexpected keys: {len(unexpected)}")
    for k in unexpected:
        print(f"  - unexpected: {k}")

    n_params = sum(v.numel() for v in sd.values())
    print(f"[load_ecapa_ckpt] saved {len(sd)} tensors / {n_params/1e6:.2f} M params → {dst}")
    torch.save(sd, dst)


if __name__ == "__main__":
    main()
