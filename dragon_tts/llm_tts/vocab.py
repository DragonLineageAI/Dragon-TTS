"""Single source of truth for the LLM-TTS token vocabulary.

The custom prompt format mixes three families of tokens, all added to the base
LLM tokenizer as *special* (non-splittable) tokens:

1. **Structural tokens (7):** mark the sections of the sequence.
2. **Speaker tokens (4096):** ``<|spk_v|>`` for v in 0..4095 — one per FSQ
   codebook entry of the speaker tokenizer (codebook_size=4096,
   num_quantizers=1). The 32 positions are emitted in sequence order.
3. **SNAC audio tokens (7*4096 = 28672):** ``<|audio_id|>`` where
   ``id = slot*4096 + value`` for ``slot`` in 0..6 (the per-frame position,
   following Orpheus's layout) and ``value`` in 0..4095 (the SNAC code). Baking
   the slot into the id makes decoding unambiguous.

All offset math lives here so encode (``snac_codec``) and decode stay in sync.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# --- structural tokens -------------------------------------------------------
TASK_TTS = "<|task_tts|>"
START_EMBEDDING = "<|start_embedding_token|>"
END_EMBEDDING = "<|end_embedding_token|>"
START_CONTENT = "<|start_content|>"
END_CONTENT = "<|end_content|>"
START_AUDIO = "<|start_audio_token|>"
END_AUDIO = "<|end_audio_token|>"

STRUCTURAL_TOKENS: List[str] = [
    TASK_TTS,
    START_EMBEDDING,
    END_EMBEDDING,
    START_CONTENT,
    END_CONTENT,
    START_AUDIO,
    END_AUDIO,
]

# --- sizes -------------------------------------------------------------------
# Speaker tokenizer: ResidualFSQ codebook_size = 4^6 = 4096, num_quantizers = 1.
SPK_CODEBOOK_SIZE = 4096
# SNAC: 3 RVQ levels flattened to 7 tokens per frame; each code is 0..4095.
SNAC_SLOTS = 7
SNAC_CODEBOOK_SIZE = 4096
NUM_AUDIO_TOKENS = SNAC_SLOTS * SNAC_CODEBOOK_SIZE  # 28672

_SPK_RE = re.compile(r"^<\|spk_(\d+)\|>$")
_AUDIO_RE = re.compile(r"^<\|audio_(\d+)\|>$")


# --- speaker tokens ----------------------------------------------------------
def spk_token(value: int) -> str:
    """``<|spk_v|>`` for an FSQ index ``value`` in 0..SPK_CODEBOOK_SIZE-1."""
    if not 0 <= value < SPK_CODEBOOK_SIZE:
        raise ValueError(f"speaker value {value} out of range [0,{SPK_CODEBOOK_SIZE})")
    return f"<|spk_{value}|>"


def parse_spk_token(token: str) -> int:
    m = _SPK_RE.match(token.strip())
    if m is None:
        raise ValueError(f"not a speaker token: {token!r}")
    return int(m.group(1))


# --- audio (SNAC) tokens -----------------------------------------------------
def audio_token(slot: int, value: int) -> str:
    """``<|audio_id|>`` with ``id = slot*4096 + value``.

    ``slot`` is the per-frame position (0..6), ``value`` the SNAC code (0..4095).
    """
    if not 0 <= slot < SNAC_SLOTS:
        raise ValueError(f"slot {slot} out of range [0,{SNAC_SLOTS})")
    if not 0 <= value < SNAC_CODEBOOK_SIZE:
        raise ValueError(f"value {value} out of range [0,{SNAC_CODEBOOK_SIZE})")
    return f"<|audio_{slot * SNAC_CODEBOOK_SIZE + value}|>"


def parse_audio_token(token: str) -> Tuple[int, int]:
    """Inverse of :func:`audio_token` → ``(slot, value)``."""
    m = _AUDIO_RE.match(token.strip())
    if m is None:
        raise ValueError(f"not an audio token: {token!r}")
    flat = int(m.group(1))
    if not 0 <= flat < NUM_AUDIO_TOKENS:
        raise ValueError(f"audio id {flat} out of range [0,{NUM_AUDIO_TOKENS})")
    return flat // SNAC_CODEBOOK_SIZE, flat % SNAC_CODEBOOK_SIZE


# --- full vocab + tokenizer --------------------------------------------------
def all_added_tokens() -> List[str]:
    """Every new token the base tokenizer must learn (~32,775 tokens)."""
    tokens = list(STRUCTURAL_TOKENS)
    tokens += [f"<|spk_{v}|>" for v in range(SPK_CODEBOOK_SIZE)]
    tokens += [f"<|audio_{i}|>" for i in range(NUM_AUDIO_TOKENS)]
    return tokens


def build_extended_tokenizer(base_model: str = "Qwen/Qwen3-0.6B"):
    """Load the base tokenizer and add all LLM-TTS special tokens.

    Returns the extended tokenizer. The model's input/output embeddings must be
    resized to ``len(tokenizer)`` afterwards (see scripts/llm_tts/build_tokenizer.py).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.add_special_tokens({"additional_special_tokens": all_added_tokens()})
    return tok
