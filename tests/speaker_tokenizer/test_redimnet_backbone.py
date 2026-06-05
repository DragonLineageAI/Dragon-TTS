"""Smoke tests cho SpeakerTokenizer với backbone ReDimNet (IDRnD/ReDimNet).

Yêu cầu network access để tải trọng số qua ``torch.hub``
(skip nếu offline / download thất bại).
"""

from __future__ import annotations

import pytest
import torch

from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer


MODEL_NAME = "M"
TRAIN_TYPE = "ft_mix"
DATASET = "vb2+vox2+cnc"


def _build_model() -> SpeakerTokenizer:
    try:
        model = SpeakerTokenizer(
            encoder={
                "type": "redimnet",
                "model_name": MODEL_NAME,
                "train_type": TRAIN_TYPE,
                "dataset": DATASET,
            },
            latent_dim=128,
            token_num=32,
            fsq_levels=[4, 4, 4, 4, 4, 4],
            fsq_num_quantizers=1,
        )
    except Exception as e:  # network / hub unavailable
        pytest.skip(f"ReDimNet weights unavailable: {e}")
    model.freeze_encoder()
    model.eval()
    return model


def test_forward_shapes_redimnet():
    model = _build_model()
    wav = torch.randn(2, 16000)  # 1 s @ 16 kHz
    x_vec, d_vec, indices = model(wav)
    assert x_vec.shape == (2, 192), x_vec.shape  # ReDimNet embedding dim
    assert d_vec.shape == (2, 192), d_vec.shape
    assert indices.shape == (2, 1, 32), indices.shape
    assert model.codebook_size == 4096
    assert model.input_kind == "waveform"


def test_round_trip_deterministic_redimnet():
    model = _build_model()
    wav = torch.randn(3, 16000)
    _, d_vec_forward, indices = model(wav)
    d_vec_detok = model.detokenize(indices)
    assert torch.allclose(d_vec_forward, d_vec_detok, atol=1e-5)


def test_encoder_frozen_redimnet():
    model = _build_model()
    n_trainable_backbone = sum(
        p.numel() for p in model.speaker_encoder.parameters() if p.requires_grad
    )
    assert n_trainable_backbone == 0, "ReDimNet backbone phải bị freeze"

    n_trainable_other = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("speaker_encoder.")
    )
    assert n_trainable_other > 0


def test_train_keeps_backbone_eval_redimnet():
    model = _build_model()
    model.train()
    # ReDimNet BatchNorm phải ở eval khi frozen → forward deterministic.
    assert not model.speaker_encoder.training
