"""SNAC neural-codec wrapper (encode/decode) for the LLM-TTS token stream.

Uses ``hubertsiuzdak/snac_24khz`` at 24 kHz. Each frame is flattened into 7
tokens following the Orpheus layout so that the upstream generator and the
downstream decoder agree on slot ordering:

    slot 0 -> codes[0][i]
    slot 1 -> codes[1][2i]      slot 4 -> codes[1][2i+1]
    slot 2 -> codes[2][4i]      slot 3 -> codes[2][4i+1]
    slot 5 -> codes[2][4i+2]    slot 6 -> codes[2][4i+3]
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from dragon_tts.llm_tts.vocab import (
    SNAC_SLOTS,
    audio_token,
    parse_audio_token,
)

SNAC_SAMPLE_RATE = 24000
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


class SnacCodec:
    """Thin wrapper around the SNAC 24 kHz model."""

    def __init__(self, device: str = "cuda", model_id: str = SNAC_MODEL_ID):
        from snac import SNAC  # lazy: heavy optional dependency

        self.device = torch.device(device)
        self.model = SNAC.from_pretrained(model_id).eval().to(self.device)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, wav_24k: torch.Tensor) -> List[str]:
        """Encode a single 24 kHz waveform into a flat list of ``<|audio_*|>``.

        ``wav_24k`` is a mono waveform of shape ``(T,)`` or ``(1, T)``.
        Returns ``7 * num_frames`` token strings.
        """
        wav = wav_24k
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B=1, 1, T)
        wav = wav.to(self.device)

        codes = self.model.encode(wav)  # list of (1, n0), (1, n1), (1, n2)
        c0, c1, c2 = codes[0][0], codes[1][0], codes[2][0]

        tokens: List[str] = []
        for i in range(c0.shape[0]):
            tokens.append(audio_token(0, int(c0[i])))
            tokens.append(audio_token(1, int(c1[2 * i])))
            tokens.append(audio_token(2, int(c2[4 * i])))
            tokens.append(audio_token(3, int(c2[4 * i + 1])))
            tokens.append(audio_token(4, int(c1[2 * i + 1])))
            tokens.append(audio_token(5, int(c2[4 * i + 2])))
            tokens.append(audio_token(6, int(c2[4 * i + 3])))
        return tokens

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, audio_tokens: Sequence[str]) -> torch.Tensor:
        """Decode ``<|audio_*|>`` token strings back to a 24 kHz waveform.

        Trailing tokens that do not complete a 7-token frame are dropped.
        Returns a 1-D CPU tensor of audio samples.
        """
        pairs: List[Tuple[int, int]] = [parse_audio_token(t) for t in audio_tokens]
        return self.decode_pairs(pairs)

    @torch.no_grad()
    def decode_pairs(self, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Decode parsed ``(slot, value)`` pairs to a 24 kHz waveform."""
        n_frames = len(pairs) // SNAC_SLOTS
        if n_frames == 0:
            return torch.zeros(0)

        codes_0: List[int] = []
        codes_1: List[int] = []
        codes_2: List[int] = []
        for j in range(n_frames):
            base = SNAC_SLOTS * j
            frame = [pairs[base + s][1] for s in range(SNAC_SLOTS)]
            codes_0.append(frame[0])
            codes_1.append(frame[1])
            codes_1.append(frame[4])
            codes_2.append(frame[2])
            codes_2.append(frame[3])
            codes_2.append(frame[5])
            codes_2.append(frame[6])

        codes = [
            torch.tensor(codes_0, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(codes_1, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(codes_2, dtype=torch.long, device=self.device).unsqueeze(0),
        ]
        audio = self.model.decode(codes)  # (1, 1, T)
        return audio.squeeze().detach().cpu()
