"""Create and save the extended Qwen3 tokenizer (+ optionally a resized model).

Adds all LLM-TTS special tokens (structural + speaker + audio) to the base
tokenizer and resizes the base model's embeddings to match, so the saved
tokenizer/model can be used directly for training and inference.

The audio token count depends on the chosen codec:
  - SNAC:     7 + 4096 + 28672 = 32775
  - NeuCodec: 7 + 4096 + 65536 = 69639

Usage:
    PYTHONPATH=. python scripts/llm_tts/build_tokenizer.py --config-name base \
        tokenizer.out_dir=./ckpts/llm_tts_tokenizer

    # NeuCodec tokenizer:
    PYTHONPATH=. python scripts/llm_tts/build_tokenizer.py --config-name base \
        codec=neucodec tokenizer.out_dir=./ckpts/llm_tts_tokenizer_neucodec

    # also save a resized base model:
    PYTHONPATH=. python scripts/llm_tts/build_tokenizer.py --config-name base \
        tokenizer.out_dir=./ckpts/llm_tts_qwen3 tokenizer.save_model=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from dragon_tts.llm_tts.vocab import (
    CodecType,
    all_added_tokens,
    build_extended_tokenizer,
)


def _resolve_codec_type(cfg: DictConfig) -> CodecType | None:
    """Map the config ``codec`` string to a :class:`CodecType`."""
    name = cfg.get("codec", "snac")
    try:
        return CodecType(name)
    except ValueError:
        return None  # defaults to SNAC in vocab helpers


def _mean_init_new_rows(model, n_added: int) -> None:
    """Initialise the newly added embedding rows with the mean of existing ones.

    Helps convergence vs. the default random init for ~33k fresh tokens.
    """
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        if emb is None:
            continue
        w = emb.weight.data
        old = w[:-n_added] if n_added > 0 else w
        mean = old.mean(dim=0, keepdim=True)
        if n_added > 0:
            w[-n_added:] = mean


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "llm_tts"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    base_model = cfg.base_model
    out_dir = Path(cfg.tokenizer.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec_type = _resolve_codec_type(cfg)
    tok = build_extended_tokenizer(base_model, codec_type=codec_type)
    tok.save_pretrained(str(out_dir))
    n_added = len(all_added_tokens(codec_type))
    print(
        f"[tokenizer] base={base_model}  codec={codec_type}  "
        f"vocab_size={len(tok)}  added={n_added}"
    )
    print(f"[tokenizer] saved to {out_dir}")

    if cfg.tokenizer.get("save_model", False):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
        model.resize_token_embeddings(len(tok))
        if cfg.tokenizer.get("mean_init_new", True):
            _mean_init_new_rows(model, n_added)
        model.save_pretrained(str(out_dir))
        print(f"[model] resized embeddings to {len(tok)} and saved to {out_dir}")


if __name__ == "__main__":
    main()
