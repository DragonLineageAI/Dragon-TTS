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
    assert seen == set(range(vocab.NUM_AUDIO_TOKENS))


def test_spk_token_roundtrip():
    for v in (0, 17, 4095):
        assert vocab.parse_spk_token(vocab.spk_token(v)) == v


def test_token_count():
    toks = vocab.all_added_tokens()
    assert len(toks) == len(set(toks))  # no duplicates
    assert len(toks) == 7 + vocab.SPK_CODEBOOK_SIZE + vocab.NUM_AUDIO_TOKENS


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
