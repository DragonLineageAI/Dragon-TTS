"""Pure round-trip tests for the LLM-TTS vocabulary + sequence assembly.

No heavy deps (no torch / transformers / snac) — just the offset math.
"""

import pytest

from dragon_tts.llm_tts import vocab
from dragon_tts.llm_tts.sequence import build_prompt, build_sequence, parse_sequence


def test_audio_token_roundtrip_all_slots():
    for slot in range(vocab.SNAC_SLOTS):
        for value in (0, 1, 2047, 4095):
            tok = vocab.audio_token(slot, value)
            assert vocab.parse_audio_token(tok) == (slot, value)


def test_audio_token_ids_are_unique_and_contiguous():
    seen = set()
    for slot in range(vocab.SNAC_SLOTS):
        for value in range(vocab.SNAC_CODEBOOK_SIZE):
            flat = int(vocab.audio_token(slot, value)[len("<|audio_") : -2])
            seen.add(flat)
    assert seen == set(range(vocab.NUM_AUDIO_TOKENS_SNAC))


def test_spk_token_roundtrip():
    for v in (0, 17, 4095):
        assert vocab.parse_spk_token(vocab.spk_token(v)) == v


def test_token_count():
    toks = vocab.all_added_tokens()
    assert len(toks) == len(set(toks))  # no duplicates
    assert len(toks) == 7 + vocab.SPK_CODEBOOK_SIZE + vocab.NUM_AUDIO_TOKENS_SNAC


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        vocab.audio_token(7, 0)
    with pytest.raises(ValueError):
        vocab.audio_token(0, 4096)
    with pytest.raises(ValueError):
        vocab.spk_token(4096)


def test_sequence_roundtrip():
    spk_ids = [0, 5, 4095, 100, 7]
    text = "Hello, world! Xin chào."
    audio_tokens = [
        vocab.audio_token(slot, (slot * 13 + 1) % 4096)
        for _ in range(3)
        for slot in range(7)
    ]
    seq = build_sequence(spk_ids, text, audio_tokens)
    parsed = parse_sequence(seq)
    assert parsed["spk_ids"] == spk_ids
    assert parsed["text"] == text
    assert parsed["audio_pairs"] == [vocab.parse_audio_token(t) for t in audio_tokens]


def test_prompt_is_sequence_prefix():
    spk_ids = [1, 2, 3]
    text = "abc"
    prompt = build_prompt(spk_ids, text)
    seq = build_sequence(spk_ids, text, [vocab.audio_token(0, 0)])
    assert seq.startswith(prompt)
    assert prompt.endswith(vocab.START_AUDIO)


# --- NeuCodec-specific tests -------------------------------------------------

def test_neucodec_audio_token_roundtrip():
    """NeuCodec: slot=0, codebook_size=65536."""
    cb = vocab.NEUCODEC_CODEBOOK_SIZE
    for value in (0, 1, 100, 32767, 65535):
        tok = vocab.audio_token(0, value, slots=1, codebook_size=cb)
        slot, val = vocab.parse_audio_token(tok, codebook_size=cb, num_audio_tokens=cb)
        assert slot == 0
        assert val == value


def test_neucodec_token_count():
    toks = vocab.all_added_tokens(vocab.CodecType.NEUCODEC)
    assert len(toks) == len(set(toks))  # no duplicates
    expected = 7 + vocab.SPK_CODEBOOK_SIZE + vocab.NUM_AUDIO_TOKENS_NEUCODEC
    assert len(toks) == expected
    assert expected == 7 + 4096 + 65536  # = 69639


def test_neucodec_out_of_range_raises():
    cb = vocab.NEUCODEC_CODEBOOK_SIZE
    with pytest.raises(ValueError):
        vocab.audio_token(1, 0, slots=1, codebook_size=cb)  # slot 1 invalid
    with pytest.raises(ValueError):
        vocab.audio_token(0, 65536, slots=1, codebook_size=cb)  # value out of range


def test_neucodec_sequence_roundtrip():
    """Build + parse a sequence using NeuCodec tokens."""
    spk_ids = [0, 5, 4095]
    text = "Xin chào!"
    cb = vocab.NEUCODEC_CODEBOOK_SIZE
    audio_tokens = [
        vocab.audio_token(0, v, slots=1, codebook_size=cb)
        for v in [0, 100, 65535, 32768, 1234]
    ]
    seq = build_sequence(spk_ids, text, audio_tokens)
    parsed = parse_sequence(seq, codebook_size=cb, num_audio_tokens=cb)
    assert parsed["spk_ids"] == spk_ids
    assert parsed["text"] == text
    expected_pairs = [
        vocab.parse_audio_token(t, codebook_size=cb, num_audio_tokens=cb)
        for t in audio_tokens
    ]
    assert parsed["audio_pairs"] == expected_pairs


def test_codec_type_enum():
    assert vocab.CodecType.SNAC.value == "snac"
    assert vocab.CodecType.NEUCODEC.value == "neucodec"
    assert vocab.CodecType("snac") == vocab.CodecType.SNAC
    assert vocab.CodecType("neucodec") == vocab.CodecType.NEUCODEC
