"""Smoke tests cho SpeakerTokenizer với backbone Qwen3 ECAPA-TDNN.

Yêu cầu `transformers` + model weights tại
`ckpts/Qwen3-Voice-Embedding-12Hz-1.7B` (skip nếu không có).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer

PRETRAINED = str(
    Path(__file__).resolve().parents[2] / "ckpts" / "Qwen3-Voice-Embedding-12Hz-1.7B"
)


def _build_model() -> SpeakerTokenizer:
    if not Path(PRETRAINED).exists():
        pytest.skip(f"Qwen3 weights not found at {PRETRAINED}")
    try:
        model = SpeakerTokenizer(
            encoder={
                "type": "qwen3",
                "pretrained": PRETRAINED,
                "sample_rate": 24000,
            },
            latent_dim=128,
            token_num=32,
            fsq_levels=[4, 4, 4, 4, 4, 4],
            fsq_num_quantizers=1,
        )
    except Exception as e:
        pytest.skip(f"Qwen3 backbone unavailable: {e}")
    model.freeze_encoder()
    model.eval()
    return model


def _preprocess(model, wav_tensor):
    """Use the backbone's processor to convert waveform → mel + mask."""
    backbone = model.speaker_encoder
    features = backbone.preprocess(wav_tensor)
    return features["input_values"], features["attention_mask"]


def test_forward_shapes_qwen3():
    model = _build_model()
    wav = torch.randn(2, 24000)  # 1 s @ 24 kHz
    mel, mask = _preprocess(model, wav)
    x_vec, d_vec, indices = model(mel, attention_mask=mask)
    assert x_vec.shape == (2, 2048), x_vec.shape  # Qwen3 embedding dim
    assert d_vec.shape == (2, 2048), d_vec.shape
    assert indices.shape == (2, 1, 32), indices.shape
    assert model.codebook_size == 4096
    assert model.input_kind == "waveform_raw"


def test_round_trip_deterministic_qwen3():
    model = _build_model()
    wav = torch.randn(3, 24000)
    mel, mask = _preprocess(model, wav)
    _, d_vec_forward, indices = model(mel, attention_mask=mask)
    d_vec_detok = model.detokenize(indices)
    assert torch.allclose(d_vec_forward, d_vec_detok, atol=1e-5)


def test_encoder_frozen_qwen3():
    model = _build_model()
    n_trainable_backbone = sum(
        p.numel() for p in model.speaker_encoder.parameters() if p.requires_grad
    )
    assert n_trainable_backbone == 0, "Qwen3 backbone phải bị freeze"

    n_trainable_other = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("speaker_encoder.")
    )
    assert n_trainable_other > 0


def test_train_keeps_backbone_eval_qwen3():
    model = _build_model()
    model.train()
    # Conv layers should stay in eval when frozen → deterministic forward.
    assert not model.speaker_encoder.training


def test_features_shape_qwen3():
    """Verify context_dim = 1536 from concatenated SE-Res2Net block outputs."""
    model = _build_model()
    wav = torch.randn(1, 24000)
    mel, mask = _preprocess(model, wav)
    backbone = model.speaker_encoder
    with torch.no_grad():
        x_vec, features = backbone(mel, attention_mask=mask)
    assert features.shape[0] == 1
    assert features.shape[1] == 1536, f"Expected context_dim=1536, got {features.shape[1]}"
    assert features.dim() == 3  # (B, 1536, T)
    assert x_vec.shape == (1, 2048)


def test_variable_length_with_mask_qwen3():
    """Variable-length batch via processor should produce valid outputs."""
    model = _build_model()
    # Two audios of different lengths
    wav_1s = torch.randn(24000)   # 1 second
    wav_05s = torch.randn(12000)  # 0.5 seconds
    # Processor handles padding + mask generation
    backbone = model.speaker_encoder
    import numpy as np
    features = backbone.preprocess(
        [wav_1s.numpy(), wav_05s.numpy()],
        sampling_rate=24000,
    )
    mel = features["input_values"]
    mask = features["attention_mask"]
    x_vec, d_vec, indices = model(mel, attention_mask=mask)
    assert x_vec.shape == (2, 2048)
    assert d_vec.shape == (2, 2048)
    assert indices.shape == (2, 1, 32)
    # Both embeddings should be finite
    assert torch.isfinite(x_vec).all()
    assert torch.isfinite(d_vec).all()
    # Mask should reflect different lengths
    assert mask.shape[0] == 2
    assert mask[0].sum() >= mask[1].sum()  # first audio is longer
