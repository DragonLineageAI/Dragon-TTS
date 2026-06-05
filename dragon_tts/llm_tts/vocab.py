"""Single source of truth for the LLM-TTS token vocabulary.

The custom prompt format mixes three families of tokens, all added to the base
LLM tokenizer as *special* (non-splittable) tokens:

1. **Structural tokens (7):** mark the sections of the sequence.
2. **Speaker tokens (4096):** ``<|spk_v|>`` for v in 0..4095 — one per FSQ
   codebook entry of the speaker tokenizer (codebook_size=4096,
   num_quantizers=1). The 32 positions are emitted in sequence order.
3. **Audio tokens:** ``<|audio_id|>`` where ``id = slot*codebook_size + value``.
   The exact token count depends on the codec:

   - **SNAC** (default): 7 slots × 4096 codebook = 28672 tokens.
   - **NeuCodec**: 1 slot × 65536 codebook = 65536 tokens.

   Baking the slot into the id makes decoding unambiguous.

All offset math lives here so encode (``snac_codec`` / ``neucodec_codec``) and
decode stay in sync.
"""

from __future__ import annotations

import enum
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

# --- codec type --------------------------------------------------------------
class CodecType(enum.Enum):
    SNAC = "snac"
    NEUCODEC = "neucodec"


# --- sizes -------------------------------------------------------------------
# Speaker tokenizer: ResidualFSQ codebook_size = 4^6 = 4096, num_quantizers = 1.
SPK_CODEBOOK_SIZE = 4096

# SNAC: 3 RVQ levels flattened to 7 tokens per frame; each code is 0..4095.
SNAC_SLOTS = 7
SNAC_CODEBOOK_SIZE = 4096
NUM_AUDIO_TOKENS_SNAC = SNAC_SLOTS * SNAC_CODEBOOK_SIZE  # 28672

# NeuCodec: FSQ with 1 token per frame; each code is 0..65535.
NEUCODEC_SLOTS = 1
NEUCODEC_CODEBOOK_SIZE = 65536
NUM_AUDIO_TOKENS_NEUCODEC = NEUCODEC_SLOTS * NEUCODEC_CODEBOOK_SIZE  # 65536

# Backward compatibility: default to SNAC.
NUM_AUDIO_TOKENS = NUM_AUDIO_TOKENS_SNAC

_SPK_RE = re.compile(r"^<\|spk_(\d+)\|>$")
_AUDIO_RE = re.compile(r"^<\|audio_(\d+)\|>$")


def _codec_params(codec_type: CodecType | None = None) -> tuple[int, int, int]:
    """Return (slots, codebook_size, num_audio_tokens) for a codec type."""
    if codec_type is None or codec_type == CodecType.SNAC:
        return SNAC_SLOTS, SNAC_CODEBOOK_SIZE, NUM_AUDIO_TOKENS_SNAC
    elif codec_type == CodecType.NEUCODEC:
        return NEUCODEC_SLOTS, NEUCODEC_CODEBOOK_SIZE, NUM_AUDIO_TOKENS_NEUCODEC
    raise ValueError(f"unknown codec type: {codec_type}")


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


# --- audio tokens (codec-generic) --------------------------------------------
def audio_token(
    slot: int,
    value: int,
    *,
    slots: int | None = None,
    codebook_size: int | None = None,
) -> str:
    """``<|audio_id|>`` with ``id = slot*codebook_size + value``.

    When ``slots`` / ``codebook_size`` are not given, defaults to SNAC constants
    for backward compatibility.
    """
    _slots = slots if slots is not None else SNAC_SLOTS
    _cb = codebook_size if codebook_size is not None else SNAC_CODEBOOK_SIZE
    if not 0 <= slot < _slots:
        raise ValueError(f"slot {slot} out of range [0,{_slots})")
    if not 0 <= value < _cb:
        raise ValueError(f"value {value} out of range [0,{_cb})")
    return f"<|audio_{slot * _cb + value}|>"


def parse_audio_token(
    token: str,
    *,
    codebook_size: int | None = None,
    num_audio_tokens: int | None = None,
) -> Tuple[int, int]:
    """Inverse of :func:`audio_token` → ``(slot, value)``.

    When ``codebook_size`` / ``num_audio_tokens`` are not given, defaults to
    SNAC constants for backward compatibility.
    """
    _cb = codebook_size if codebook_size is not None else SNAC_CODEBOOK_SIZE
    _nat = num_audio_tokens if num_audio_tokens is not None else (_cb * SNAC_SLOTS)
    m = _AUDIO_RE.match(token.strip())
    if m is None:
        raise ValueError(f"not an audio token: {token!r}")
    flat = int(m.group(1))
    if not 0 <= flat < _nat:
        raise ValueError(f"audio id {flat} out of range [0,{_nat})")
    return flat // _cb, flat % _cb


# --- full vocab + tokenizer --------------------------------------------------
def all_added_tokens(codec_type: CodecType | None = None) -> List[str]:
    """Every new token the base tokenizer must learn.

    - SNAC: 7 + 4096 + 28672 = 32775 tokens
    - NeuCodec: 7 + 4096 + 65536 = 69639 tokens
    """
    _, _, num_audio = _codec_params(codec_type)
    tokens = list(STRUCTURAL_TOKENS)
    tokens += [f"<|spk_{v}|>" for v in range(SPK_CODEBOOK_SIZE)]
    tokens += [f"<|audio_{i}|>" for i in range(num_audio)]
    return tokens


def build_extended_tokenizer(
    base_model: str = "Qwen/Qwen3-0.6B",
    codec_type: CodecType | None = None,
):
    """Load the base tokenizer and add all LLM-TTS special tokens.

    Returns the extended tokenizer. The model's input/output embeddings must be
    resized to ``len(tokenizer)`` afterwards (see scripts/llm_tts/build_tokenizer.py).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.add_special_tokens(
        {"additional_special_tokens": all_added_tokens(codec_type)}
    )
    return tok
