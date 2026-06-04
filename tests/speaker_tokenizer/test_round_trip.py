"""Smoke tests cho SpeakerTokenizer.

- Shape của forward / tokenize / detokenize.
- tokenize → detokenize trùng với d_vector trong forward (deterministic).
- Số params trainable == 0 cho ECAPA sau khi freeze.
"""

from __future__ import annotations

import torch

from dragon_tts.speaker_tokenizer.model import SpeakerTokenizer


def _build_model(token_num: int = 32, num_quantizers: int = 1) -> SpeakerTokenizer:
    model = SpeakerTokenizer(
        input_dim=128,
        out_dim=1024,
        latent_dim=128,
        token_num=token_num,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=num_quantizers,
    )
    model.freeze_ecapa()
    model.eval()
    return model


def test_forward_shapes():
    model = _build_model()
    mel = torch.randn(2, 128, 200)
    x_vec, d_vec, indices = model(mel)
    assert x_vec.shape == (2, 1024), x_vec.shape
    assert d_vec.shape == (2, 1024), d_vec.shape
    # ResidualFSQ với is_channel_first=True trả indices (B, num_quantizers, token_num),
    # khớp với convention của SparkVox SpeakerEncoder.
    assert indices.shape == (2, 1, 32), indices.shape
    assert indices.dtype in (torch.int32, torch.int64), indices.dtype


def test_codebook_size():
    model = _build_model()
    assert model.codebook_size == 4 ** 6 == 4096


def test_round_trip_deterministic():
    model = _build_model()
    mel = torch.randn(3, 128, 200)
    _, d_vec_forward, indices = model(mel)
    d_vec_detok = model.detokenize(indices)
    assert torch.allclose(d_vec_forward, d_vec_detok, atol=1e-5), (
        "tokenize → detokenize phải trùng với d_vec trong forward"
    )


def test_tokenize_matches_forward_indices():
    model = _build_model()
    mel = torch.randn(2, 128, 200)
    _, _, indices_fwd = model(mel)
    indices_tok = model.tokenize(mel)
    assert torch.equal(indices_fwd, indices_tok)


def test_ecapa_frozen():
    model = _build_model()
    n_trainable_ecapa = sum(
        p.numel() for p in model.speaker_encoder.parameters() if p.requires_grad
    )
    assert n_trainable_ecapa == 0, "ECAPA parameters phải bị freeze sau freeze_ecapa()"

    # Đảm bảo phần còn lại vẫn trainable.
    n_trainable_other = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("speaker_encoder.")
    )
    assert n_trainable_other > 0


def test_train_eval_keeps_ecapa_eval():
    model = _build_model()
    model.train()
    # ECAPA phải vẫn ở eval mode khi parent ở train mode (BatchNorm stable).
    assert not model.speaker_encoder.training
    model.eval()
    assert not model.speaker_encoder.training
