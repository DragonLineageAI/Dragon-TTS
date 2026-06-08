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

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.vocab import (
    SNAC_SLOTS,
    audio_token,
    parse_audio_token,
)

SNAC_SAMPLE_RATE = 24000
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


class SnacCodec(AudioCodec):
    """Thin wrapper around the SNAC 24 kHz model."""

    sample_rate: int = SNAC_SAMPLE_RATE
    output_sample_rate: int = SNAC_SAMPLE_RATE
    slots_per_frame: int = SNAC_SLOTS
    codebook_size: int = 4096

    def __init__(self, device: str = "cuda", model_id: str = SNAC_MODEL_ID):
        from snac import SNAC  # lazy: heavy optional dependency

        self.device = torch.device(device)
        self.model = SNAC.from_pretrained(model_id).eval().to(self.device)
        self._hop: float | None = None  # probed lazily on first batch_encode

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
        c0 = codes[0][0].cpu().tolist()
        c1 = codes[1][0].cpu().tolist()
        c2 = codes[2][0].cpu().tolist()

        tokens: List[str] = []
        for i in range(len(c0)):
            tokens.append(audio_token(0, c0[i]))
            tokens.append(audio_token(1, c1[2 * i]))
            tokens.append(audio_token(2, c2[4 * i]))
            tokens.append(audio_token(3, c2[4 * i + 1]))
            tokens.append(audio_token(4, c1[2 * i + 1]))
            tokens.append(audio_token(5, c2[4 * i + 2]))
            tokens.append(audio_token(6, c2[4 * i + 3]))
        return tokens

    @torch.no_grad()
    def batch_encode(self, wavs_24k: List[torch.Tensor]) -> List[List[str]]:
        """Batch-encode multiple 24 kHz waveforms into lists of audio tokens.

        Pads all waveforms to the longest in the batch, encodes in a single
        forward pass, then truncates each sample's codes to the valid
        (non-padded) region.

        Each ``wavs_24k[i]`` is a 1-D tensor ``(T_i,)``.
        Returns a list (length B) of token-string lists.
        """
        if not wavs_24k:
            return []
        if len(wavs_24k) == 1:
            return [self.encode(wavs_24k[0])]

        # -- probe hop size once ------------------------------------------------
        if self._hop is None:
            self._hop = self._probe_hop()

        # -- pad & stack --------------------------------------------------------
        orig_lengths = [w.numel() for w in wavs_24k]
        max_len = max(orig_lengths)

        padded = []
        for w in wavs_24k:
            flat = w.view(-1)
            if flat.shape[0] < max_len:
                flat = torch.nn.functional.pad(flat, (0, max_len - flat.shape[0]))
            padded.append(flat)

        batch = torch.stack(padded).unsqueeze(1).to(self.device)  # (B, 1, T)

        codes = self.model.encode(batch)  # [(B, n0), (B, n1), (B, n2)]
        max_frames = codes[0].shape[1]  # frames for the padded length

        # -- per-sample: truncate to valid frames & flatten ---------------------
        # Single bulk CUDA→CPU transfer for the entire batch
        c0_all = codes[0].cpu().tolist()  # (B, n0)
        c1_all = codes[1].cpu().tolist()  # (B, n1)
        c2_all = codes[2].cpu().tolist()  # (B, n2)

        results: List[List[str]] = []
        for b in range(len(wavs_24k)):
            n_frames = min(int(orig_lengths[b] / self._hop), max_frames)
            c0 = c0_all[b]
            c1 = c1_all[b]
            c2 = c2_all[b]

            tokens: List[str] = []
            for i in range(n_frames):
                tokens.append(audio_token(0, c0[i]))
                tokens.append(audio_token(1, c1[2 * i]))
                tokens.append(audio_token(2, c2[4 * i]))
                tokens.append(audio_token(3, c2[4 * i + 1]))
                tokens.append(audio_token(4, c1[2 * i + 1]))
                tokens.append(audio_token(5, c2[4 * i + 2]))
                tokens.append(audio_token(6, c2[4 * i + 3]))
            results.append(tokens)

        return results

    def _probe_hop(self) -> float:
        """Probe the effective hop size (input samples per coarsest frame).

        Called once and cached in ``self._hop``.
        """
        probe_len = SNAC_SAMPLE_RATE * 4  # 4 seconds – long enough for accuracy
        dummy = torch.zeros(1, 1, probe_len, device=self.device)
        with torch.no_grad():
            c = self.model.encode(dummy)
        n_frames = c[0].shape[1]
        return probe_len / n_frames

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
