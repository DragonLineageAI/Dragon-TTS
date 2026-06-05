"""Synthesize speech with the LLM-TTS (vLLM streaming backend).

Usage:
    PYTHONPATH=. python scripts/llm_tts/infer_vllm.py --config-name base \
        model_dir=./ckpts/llm_tts_qwen3 \
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
        infer.text="Hello world" infer.ref_audio=/path/ref.wav \
        infer.out=./out.wav

    # Use NeuCodec:
    PYTHONPATH=. python scripts/llm_tts/infer_vllm.py --config-name base \
        codec=neucodec \
        model_dir=./ckpts/llm_tts_qwen3 \
        speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
        infer.text="Hello world" infer.ref_audio=/path/ref.wav \
        infer.out=./out.wav
"""

from __future__ import annotations

from pathlib import Path

import hydra
import soundfile as sf
from omegaconf import DictConfig

from dragon_tts.llm_tts.inference.vllm_engine import OrpheusVllmPipeline


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs" / "llm_tts"),
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    assert cfg.get("model_dir"), "Set model_dir=<fine-tuned LLM + tokenizer dir>"
    assert cfg.get("speaker_ckpt"), "Set speaker_ckpt=<path to .ckpt>"
    assert cfg.infer.get("text"), "Set infer.text=..."
    assert cfg.infer.get("ref_audio"), "Set infer.ref_audio=<reference clip>"

    vcfg = cfg.get("vllm", {})
    pipe = OrpheusVllmPipeline(
        model_dir=cfg.model_dir,
        speaker_ckpt=cfg.speaker_ckpt,
        device=cfg.device,
        codec=cfg.get("codec", "snac"),
        snac_model=cfg.get("snac_model"),
        neucodec_model=cfg.get("neucodec_model"),
        speaker_crop_seconds=cfg.get("speaker_crop_seconds", 4.0),
        dtype=vcfg.get("dtype", "bfloat16"),
        max_model_len=vcfg.get("max_model_len", 4096),
        gpu_memory_utilization=vcfg.get("gpu_memory_utilization", 0.5),
    )
    wav = pipe.synthesize(
        text=cfg.infer.text,
        ref_audio=cfg.infer.ref_audio,
        max_new_tokens=cfg.infer.get("max_new_tokens", 2000),
        temperature=cfg.infer.get("temperature", 0.6),
        top_p=cfg.infer.get("top_p", 0.95),
        repetition_penalty=cfg.infer.get("repetition_penalty", 1.1),
    )
    out_sr = pipe.codec.output_sample_rate
    out = cfg.infer.get("out", "out.wav")
    sf.write(out, wav.numpy(), out_sr)
    print(f"[done] wrote {wav.shape[0]} samples ({wav.shape[0]/out_sr:.2f}s) -> {out}")


if __name__ == "__main__":
    main()
