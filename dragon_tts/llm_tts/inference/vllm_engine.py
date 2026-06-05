"""vLLM streaming backend for the LLM-TTS, mirroring Orpheus's engine_class.py.

Uses ``AsyncLLMEngine`` to stream generated tokens and decodes them with SNAC in
small overlapping windows (4 frames) for low latency, taking the middle slice of
each decoded window for continuity — the same trick Orpheus uses in decoder.py.

vLLM is an optional, heavy dependency and is imported lazily.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple, Union

import torch
import torchaudio

from dragon_tts.llm_tts.sequence import build_prompt
from dragon_tts.llm_tts.snac_codec import SnacCodec
from dragon_tts.llm_tts.vocab import END_AUDIO, parse_audio_token
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

AudioInput = Union[str, Path, torch.Tensor]
SPEAKER_SR = 16000

# Streaming decode window: 4 frames = 28 tokens; emit the middle 2048 samples.
_WINDOW_TOKENS = 28
_SLICE_START, _SLICE_END = 2048, 4096


class OrpheusVllmPipeline:
    def __init__(
        self,
        model_dir: Union[str, Path],
        speaker_ckpt: Union[str, Path],
        device: str = "cuda",
        snac_model: Optional[str] = None,
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
        self.snac = SnacCodec(device=device, model_id=snac_model or "hubertsiuzdak/snac_24khz")
        self.speaker = SpeakerTokenizerPipeline(str(speaker_ckpt), device=device)
        self.speaker_crop = int(speaker_crop_seconds * SPEAKER_SR)
        self.end_audio_id = self.tokenizer.convert_tokens_to_ids(END_AUDIO)

    # ------------------------------------------------------------------
    def _load_ref_16k(self, ref: AudioInput) -> torch.Tensor:
        if isinstance(ref, (str, Path)):
            wav, sr = torchaudio.load(str(ref))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
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
        """Yield 24 kHz audio chunks (1-D float tensors) as tokens stream in."""
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
                    buffer.append(parse_audio_token(tok))
                except ValueError:
                    continue
                count += 1
                if count % 7 == 0 and count >= _WINDOW_TOKENS:
                    window = buffer[-_WINDOW_TOKENS:]
                    audio = self.snac.decode_pairs(window)
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
