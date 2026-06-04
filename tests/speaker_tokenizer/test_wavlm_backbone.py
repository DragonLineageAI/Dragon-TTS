"""Smoke tests cho SpeakerTokenizer với backbone WavLM (wavlm-base-plus-sv).

Yêu cầu `transformers` + tải được trọng số `microsoft/wavlm-base-plus-sv`
(skip nếu offline / không cài transformers).
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer

PRETRAINED = "microsoft/wavlm-base-plus-sv"


def _build_model() -> SpeakerTokenizer:
    try:
        model = SpeakerTokenizer(
            encoder={
                "type": "wavlm",
                "pretrained": PRETRAINED,
                "do_normalize": True,
            },
            latent_dim=128,
            token_num=32,
            fsq_levels=[4, 4, 4, 4, 4, 4],
            fsq_num_quantizers=1,
        )
    except Exception as e:  # network / hub unavailable
        pytest.skip(f"WavLM weights unavailable: {e}")
    model.freeze_encoder()
    model.eval()
    return model


def test_forward_shapes_wavlm():
    model = _build_model()
    wav = torch.randn(2, 16000)  # 1 s @ 16 kHz
    x_vec, d_vec, indices = model(wav)
    assert x_vec.shape == (2, 512), x_vec.shape  # WavLM xvector embedding dim
    assert d_vec.shape == (2, 512), d_vec.shape
    assert indices.shape == (2, 1, 32), indices.shape
    assert model.codebook_size == 4096
    assert model.input_kind == "waveform"


def test_round_trip_deterministic_wavlm():
    model = _build_model()
    wav = torch.randn(3, 16000)
    _, d_vec_forward, indices = model(wav)
    d_vec_detok = model.detokenize(indices)
    assert torch.allclose(d_vec_forward, d_vec_detok, atol=1e-5)


def test_encoder_frozen_wavlm():
    model = _build_model()
    n_trainable_backbone = sum(
        p.numel() for p in model.speaker_encoder.parameters() if p.requires_grad
    )
    assert n_trainable_backbone == 0, "WavLM backbone phải bị freeze"

    n_trainable_other = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("speaker_encoder.")
    )
    assert n_trainable_other > 0


def test_train_keeps_backbone_eval_wavlm():
    model = _build_model()
    model.train()
    # WavLM dropout/LayerNorm phải ở eval khi frozen → forward deterministic.
    assert not model.speaker_encoder.training


def test_single_forward_hidden_state_matches():
    """out.hidden_states[-1] phải trùng wavlm(...).last_hidden_state (single forward)."""
    model = _build_model()
    wav = torch.randn(1, 16000)
    backbone = model.speaker_encoder
    with torch.no_grad():
        norm = backbone._normalize(wav) if backbone.do_normalize else wav
        out = backbone.model(input_values=norm, output_hidden_states=True)
        last_from_base = backbone.model.wavlm(norm).last_hidden_state
    assert torch.allclose(out.hidden_states[-1], last_from_base, atol=1e-4)
