"""SNAC glue-logic tests.

The flatten (encode) / regroup (decode) slot mapping is validated against a fake
SNAC model so the test needs no model download. A second, optional test runs the
real ``hubertsiuzdak/snac_24khz`` codec if the ``snac`` package is installed.
"""

import pytest

torch = pytest.importorskip("torch")

from dragon_tts.llm_tts.snac_codec import SnacCodec
from dragon_tts.llm_tts.vocab import parse_audio_token


class _FakeSnac:
    """Captures the codes passed to decode; returns fixed-shape audio."""

    def __init__(self):
        self.last_codes = None

    def encode(self, wav):
        return self._codes

    def decode(self, codes):
        self.last_codes = codes
        return torch.zeros(1, 1, 16)


def _make_codec(fake):
    codec = SnacCodec.__new__(SnacCodec)  # bypass heavy __init__
    codec.device = torch.device("cpu")
    codec.model = fake
    return codec


def test_flatten_regroup_is_consistent():
    n = 3  # frames
    c0 = torch.arange(0, n).unsqueeze(0)
    c1 = torch.arange(100, 100 + 2 * n).unsqueeze(0)
    c2 = torch.arange(1000, 1000 + 4 * n).unsqueeze(0)
    fake = _FakeSnac()
    fake._codes = [c0, c1, c2]
    codec = _make_codec(fake)

    tokens = codec.encode(torch.zeros(2400))
    assert len(tokens) == 7 * n

    pairs = [parse_audio_token(t) for t in tokens]
    codec.decode_pairs(pairs)

    out0, out1, out2 = fake.last_codes
    assert torch.equal(out0, c0 % 4096)
    assert torch.equal(out1, c1 % 4096)
    assert torch.equal(out2, c2 % 4096)


def test_decode_drops_partial_frame():
    fake = _FakeSnac()
    codec = _make_codec(fake)
    # 9 pairs -> 1 complete frame (7), 2 dropped.
    pairs = [(i % 7, i) for i in range(9)]
    codec.decode_pairs(pairs)
    assert fake.last_codes[0].shape[1] == 1  # exactly one frame decoded


@pytest.mark.slow
def test_real_snac_roundtrip():
    pytest.importorskip("snac")
    try:
        codec = SnacCodec(device="cpu")
    except Exception as e:  # network / weights unavailable
        pytest.skip(f"cannot load SNAC model: {e}")
    wav = torch.randn(24000)  # 1s @ 24 kHz
    tokens = codec.encode(wav)
    assert len(tokens) % 7 == 0 and len(tokens) > 0
    audio = codec.decode(tokens)
    assert audio.numel() > 0
