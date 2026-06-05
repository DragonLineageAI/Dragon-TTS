"""NeuCodec glue-logic tests.

The flatten (encode) / regroup (decode) mapping is validated against a fake
NeuCodec model so the test needs no model download. A second, optional test
runs the real ``neuphonic/neucodec`` codec if the ``neucodec`` package is
installed.
"""

import pytest

torch = pytest.importorskip("torch")

from dragon_tts.llm_tts.neucodec_codec import NeucodecCodec
from dragon_tts.llm_tts.vocab import (
    NEUCODEC_CODEBOOK_SIZE,
    parse_audio_token,
)


class _FakeNeuCodec:
    """Captures the codes passed to decode_code; returns fixed-shape audio."""

    def __init__(self):
        self.last_codes = None

    def encode_code(self, wav):
        return self._codes

    def decode_code(self, codes):
        self.last_codes = codes
        return torch.zeros(1, 1, 480)  # ~20ms at 24 kHz


def _make_codec(fake):
    codec = NeucodecCodec.__new__(NeucodecCodec)  # bypass heavy __init__
    codec.device = torch.device("cpu")
    codec.model = fake
    return codec


def test_flatten_regroup_is_consistent():
    n = 5  # frames
    codes = torch.arange(0, n).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    fake = _FakeNeuCodec()
    fake._codes = codes
    codec = _make_codec(fake)

    tokens = codec.encode(torch.zeros(16000))
    assert len(tokens) == n  # 1 token per frame

    pairs = [
        parse_audio_token(t, codebook_size=NEUCODEC_CODEBOOK_SIZE)
        for t in tokens
    ]
    codec.decode_pairs(pairs)

    # Verify codes passed to decode_code match the original
    out = fake.last_codes  # (1, 1, N)
    assert out.shape == (1, 1, n)
    assert torch.equal(out[0, 0], codes[0, 0])


def test_decode_drops_partial_frame():
    """NeuCodec has 1 token/frame, so every token is a complete frame."""
    fake = _FakeNeuCodec()
    codec = _make_codec(fake)
    # 3 pairs → 3 complete frames (slots_per_frame=1, so no partial)
    pairs = [(0, i) for i in range(3)]
    codec.decode_pairs(pairs)
    assert fake.last_codes.shape[2] == 3


def test_empty_pairs_returns_empty():
    fake = _FakeNeuCodec()
    codec = _make_codec(fake)
    result = codec.decode_pairs([])
    assert result.numel() == 0


def test_token_values_in_range():
    """Ensure encoded token values stay within NeuCodec codebook range."""
    n = 3
    # Use values within codebook range
    values = torch.tensor([0, 100, 65535])
    codes = values.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    fake = _FakeNeuCodec()
    fake._codes = codes
    codec = _make_codec(fake)

    tokens = codec.encode(torch.zeros(16000))
    assert len(tokens) == n

    for t in tokens:
        slot, val = parse_audio_token(
            t, codebook_size=NEUCODEC_CODEBOOK_SIZE
        )
        assert slot == 0  # NeuCodec always uses slot 0
        assert 0 <= val < NEUCODEC_CODEBOOK_SIZE


@pytest.mark.slow
def test_real_neucodec_roundtrip():
    pytest.importorskip("neucodec")
    try:
        codec = NeucodecCodec(device="cpu")
    except Exception as e:  # network / weights unavailable
        pytest.skip(f"cannot load NeuCodec model: {e}")
    wav = torch.randn(16000)  # 1s @ 16 kHz
    tokens = codec.encode(wav)
    assert len(tokens) > 0
    audio = codec.decode(tokens)
    assert audio.numel() > 0
