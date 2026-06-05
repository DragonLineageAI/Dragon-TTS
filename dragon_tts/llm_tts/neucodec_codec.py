"""NeuCodec neural-codec wrapper (encode/decode) for the LLM-TTS token stream.

Uses ``neuphonic/neucodec`` — a Finite Scalar Quantisation (FSQ) codec that
operates at 50 tokens/sec with a single codebook of 65536 entries (2^16).

Key differences from SNAC:
  - Input: 16 kHz  (SNAC: 24 kHz)
  - Output: 24 kHz (same as SNAC)
  - 1 token per frame  (SNAC: 7 tokens per frame)
  - Codebook size: 65536  (SNAC: 4096)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.vocab import audio_token, parse_audio_token

NEUCODEC_SAMPLE_RATE = 16000
NEUCODEC_OUTPUT_SAMPLE_RATE = 24000
NEUCODEC_MODEL_ID = "neuphonic/neucodec"


class NeucodecCodec(AudioCodec):
    """Thin wrapper around the NeuCodec FSQ model."""

    sample_rate: int = NEUCODEC_SAMPLE_RATE
    output_sample_rate: int = NEUCODEC_OUTPUT_SAMPLE_RATE
    slots_per_frame: int = 1
    codebook_size: int = 65536

    def __init__(self, device: str = "cuda", model_id: str = NEUCODEC_MODEL_ID):
        from neucodec import NeuCodec  # lazy: heavy optional dependency

        self.device = torch.device(device)
        self.model = NeuCodec.from_pretrained(model_id).eval().to(self.device)
        self._hop: float | None = None  # probed lazily on first batch_encode

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, wav_16k: torch.Tensor) -> List[str]:
        """Encode a single 16 kHz waveform into a flat list of ``<|audio_*|>``.

        ``wav_16k`` is a mono waveform of shape ``(T,)`` or ``(1, T)``.
        Returns ``num_frames`` token strings (1 token per frame).
        """
        wav = wav_16k
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)  # (B=1, 1, T)
        wav = wav.to(self.device)

        # encode_code returns (B, 1, N) where N = number of frames
        fsq_codes = self.model.encode_code(wav)  # (1, 1, N)
        codes = fsq_codes[0, 0]  # (N,)

        tokens: List[str] = []
        for i in range(codes.shape[0]):
            tokens.append(audio_token(0, int(codes[i]), codebook_size=self.codebook_size))
        return tokens

    @torch.no_grad()
    def batch_encode(self, wavs_16k: List[torch.Tensor]) -> List[List[str]]:
        """Batch-encode multiple 16 kHz waveforms into lists of audio tokens.

        Pads all waveforms to the longest in the batch, encodes in a single
        forward pass, then truncates each sample's codes to the valid
        (non-padded) region.

        Each ``wavs_16k[i]`` is a 1-D tensor ``(T_i,)``.
        Returns a list (length B) of token-string lists.
        """
        if not wavs_16k:
            return []
        if len(wavs_16k) == 1:
            return [self.encode(wavs_16k[0])]

        # -- probe hop size once ------------------------------------------------
        if self._hop is None:
            self._hop = self._probe_hop()

        # -- pad & stack --------------------------------------------------------
        orig_lengths = [w.numel() for w in wavs_16k]
        max_len = max(orig_lengths)

        padded = []
        for w in wavs_16k:
            flat = w.view(-1)
            if flat.shape[0] < max_len:
                flat = torch.nn.functional.pad(flat, (0, max_len - flat.shape[0]))
            padded.append(flat)

        batch = torch.stack(padded).unsqueeze(1).to(self.device)  # (B, 1, T)

        fsq_codes = self.model.encode_code(batch)  # (B, 1, N)
        max_frames = fsq_codes.shape[2]

        # -- per-sample: truncate to valid frames & flatten ---------------------
        results: List[List[str]] = []
        for b in range(len(wavs_16k)):
            n_frames = min(int(orig_lengths[b] / self._hop), max_frames)
            codes = fsq_codes[b, 0]  # (N,)

            tokens: List[str] = []
            for i in range(n_frames):
                tokens.append(
                    audio_token(0, int(codes[i]), codebook_size=self.codebook_size)
                )
            results.append(tokens)

        return results

    def _probe_hop(self) -> float:
        """Probe the effective hop size (input samples per frame).

        Called once and cached in ``self._hop``.
        """
        probe_len = NEUCODEC_SAMPLE_RATE * 4  # 4 seconds
        dummy = torch.zeros(1, 1, probe_len, device=self.device)
        with torch.no_grad():
            c = self.model.encode_code(dummy)
        n_frames = c.shape[2]
        return probe_len / n_frames

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, audio_tokens: Sequence[str]) -> torch.Tensor:
        """Decode ``<|audio_*|>`` token strings back to a 24 kHz waveform.

        Returns a 1-D CPU tensor of audio samples.
        """
        pairs: List[Tuple[int, int]] = [
            parse_audio_token(t, codebook_size=self.codebook_size)
            for t in audio_tokens
        ]
        return self.decode_pairs(pairs)

    @torch.no_grad()
    def decode_pairs(self, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Decode parsed ``(slot, value)`` pairs to a 24 kHz waveform."""
        n_frames = len(pairs) // self.slots_per_frame
        if n_frames == 0:
            return torch.zeros(0)

        # NeuCodec: 1 token per frame, slot is always 0
        values = [pairs[i][1] for i in range(n_frames)]
        codes = (
            torch.tensor(values, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, N)

        audio = self.model.decode_code(codes)  # (1, 1, T_24k)
        return audio.squeeze().detach().cpu()
