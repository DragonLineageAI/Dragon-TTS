"""Assemble / parse the LLM-TTS training and inference sequences.

Format::

    <|task_tts|><|start_embedding_token|>{spk}<|end_embedding_token|>
    <|start_content|>{text}<|end_content|>
    <|start_audio_token|>{audio}<|end_audio_token|>

(no whitespace between tokens in the actual string).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from dragon_tts.llm_tts.vocab import (
    END_AUDIO,
    END_CONTENT,
    END_EMBEDDING,
    START_AUDIO,
    START_CONTENT,
    START_EMBEDDING,
    TASK_TTS,
    parse_audio_token,
    parse_spk_token,
    spk_token,
)


def build_sequence(
    spk_ids: Sequence[int], text: str, audio_tokens: Sequence[str]
) -> str:
    """Full training example: speaker tokens + text + audio tokens."""
    spk = "".join(spk_token(int(v)) for v in spk_ids)
    audio = "".join(audio_tokens)
    return (
        f"{TASK_TTS}"
        f"{START_EMBEDDING}{spk}{END_EMBEDDING}"
        f"{START_CONTENT}{text}{END_CONTENT}"
        f"{START_AUDIO}{audio}{END_AUDIO}"
    )


def build_prompt(spk_ids: Sequence[int], text: str) -> str:
    """Inference prompt: everything up to (and including) ``<|start_audio_token|>``.

    The LLM then generates the audio tokens followed by ``<|end_audio_token|>``.
    """
    spk = "".join(spk_token(int(v)) for v in spk_ids)
    return (
        f"{TASK_TTS}"
        f"{START_EMBEDDING}{spk}{END_EMBEDDING}"
        f"{START_CONTENT}{text}{END_CONTENT}"
        f"{START_AUDIO}"
    )


_SECTION_RE = re.compile(
    re.escape(START_EMBEDDING) + r"(?P<spk>.*?)" + re.escape(END_EMBEDDING)
    + r".*?"
    + re.escape(START_CONTENT) + r"(?P<text>.*?)" + re.escape(END_CONTENT)
    + r".*?"
    + re.escape(START_AUDIO) + r"(?P<audio>.*?)" + re.escape(END_AUDIO),
    re.DOTALL,
)
_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def parse_sequence(seq: str) -> Dict[str, object]:
    """Inverse of :func:`build_sequence`.

    Returns ``{"spk_ids": List[int], "text": str, "audio_pairs": List[(slot,value)]}``.
    """
    m = _SECTION_RE.search(seq)
    if m is None:
        raise ValueError("sequence does not match the expected LLM-TTS format")
    spk_ids = [parse_spk_token(t) for t in _TOKEN_RE.findall(m.group("spk"))]
    audio_pairs: List[Tuple[int, int]] = [
        parse_audio_token(t) for t in _TOKEN_RE.findall(m.group("audio"))
    ]
    return {"spk_ids": spk_ids, "text": m.group("text"), "audio_pairs": audio_pairs}
