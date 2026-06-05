"""vLLM streaming backend for the LLM-TTS, mirroring Orpheus's engine_class.py.

Uses ``AsyncLLMEngine`` to stream generated tokens and decodes them with the
selected codec in small overlapping windows for low latency, taking the middle
slice of each decoded window for continuity — the same trick Orpheus uses.

Supports both SNAC (default) and NeuCodec codecs.

vLLM is an optional, heavy dependency and is imported lazily.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.sequence import build_prompt
from dragon_tts.llm_tts.vocab import END_AUDIO, parse_audio_token
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

AudioInput = Union[str, Path, torch.Tensor]
SPEAKER_SR = 16000

# Streaming decode window defaults per codec (overridable).
# SNAC: 4 frames × 7 tokens = 28 tokens; NeuCodec: 20 frames × 1 = 20 tokens.
_SNAC_WINDOW_FRAMES = 4
_NEUCODEC_WINDOW_FRAMES = 20
_SLICE_START, _SLICE_END = 2048, 4096


def _build_codec(
    codec_name: str, device: str, snac_model: str | None, neucodec_model: str | None
) -> AudioCodec:
    """Instantiate the configured audio codec."""
    if codec_name == "neucodec":
        from dragon_tts.llm_tts.neucodec_codec import NeucodecCodec

        return NeucodecCodec(
            device=device, model_id=neucodec_model or "neuphonic/neucodec"
        )
    else:
        from dragon_tts.llm_tts.snac_codec import SnacCodec

        return SnacCodec(
            device=device, model_id=snac_model or "hubertsiuzdak/snac_24khz"
        )


class OrpheusVllmPipeline:
    def __init__(
        self,
        model_dir: Union[str, Path],
        speaker_ckpt: Union[str, Path],
        device: str = "cuda",
        codec: str = "snac",
        snac_model: Optional[str] = None,
        neucodec_model: Optional[str] = None,
        speaker_crop_seconds: float = 4.0,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.5,
    ):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        engine_args = AsyncEngineArgs(
            model=str(model_dir),
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.codec: AudioCodec = _build_codec(codec, device, snac_model, neucodec_model)
        self.speaker = SpeakerTokenizerPipeline(str(speaker_ckpt), device=device)
        self.speaker_crop = int(speaker_crop_seconds * SPEAKER_SR)
        self.end_audio_id = self.tokenizer.convert_tokens_to_ids(END_AUDIO)

        # Streaming decode window: tokens per window
        slots = self.codec.slots_per_frame
        if codec == "neucodec":
            self._window_tokens = _NEUCODEC_WINDOW_FRAMES * slots
        else:
            self._window_tokens = _SNAC_WINDOW_FRAMES * slots

    # ------------------------------------------------------------------
    def _load_ref_16k(self, ref: AudioInput) -> torch.Tensor:
        if isinstance(ref, (str, Path)):
            data, sr = sf.read(str(ref), dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data)
            if sr != SPEAKER_SR:
                wav = torchaudio.functional.resample(wav, sr, SPEAKER_SR)
        else:
            wav = ref.squeeze()
        n = wav.shape[0]
        target = self.speaker_crop
        if n > target:
            start = (n - target) // 2
            wav = wav[start : start + target]
        elif 0 < n < target:
            reps = -(-target // n)
            wav = wav.repeat(reps)[:target]
        return wav

    def speaker_tokens(self, ref: AudioInput) -> List[int]:
        wav = self._load_ref_16k(ref).unsqueeze(0)
        return self.speaker.tokenize(wav)[0, 0, :].cpu().tolist()

    # ------------------------------------------------------------------
    async def stream(
        self,
        text: str,
        ref_audio: AudioInput,
        max_new_tokens: int = 2000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> AsyncGenerator[torch.Tensor, None]:
        """Yield audio chunks (1-D float tensors) as tokens stream in."""
        from vllm import SamplingParams

        spk_ids = self.speaker_tokens(ref_audio)
        prompt = build_prompt(spk_ids, text)
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[self.end_audio_id],
        )
        request_id = str(uuid.uuid4())
        results = self.engine.generate(prompt, sampling, request_id)

        cb = self.codec.codebook_size
        nat = self.codec.num_audio_tokens
        slots = self.codec.slots_per_frame
        window_tokens = self._window_tokens

        buffer: List[Tuple[int, int]] = []
        count = 0
        prev_len = 0
        async for out in results:
            token_ids = out.outputs[0].token_ids
            for tid in token_ids[prev_len:]:
                if tid == self.end_audio_id:
                    return
                tok = self.tokenizer.convert_ids_to_tokens(int(tid))
                try:
                    buffer.append(
                        parse_audio_token(tok, codebook_size=cb, num_audio_tokens=nat)
                    )
                except ValueError:
                    continue
                count += 1
                if count % slots == 0 and count >= window_tokens:
                    window = buffer[-window_tokens:]
                    audio = self.codec.decode_pairs(window)
                    if audio.numel() >= _SLICE_END:
                        yield audio[_SLICE_START:_SLICE_END]
            prev_len = len(token_ids)

    # ------------------------------------------------------------------
    def synthesize(self, text: str, ref_audio: AudioInput, **kwargs) -> torch.Tensor:
        """Blocking convenience wrapper: collect the stream into one waveform."""

        async def _collect() -> torch.Tensor:
            chunks = [c async for c in self.stream(text, ref_audio, **kwargs)]
            return torch.cat(chunks) if chunks else torch.zeros(0)

        return asyncio.run(_collect())
