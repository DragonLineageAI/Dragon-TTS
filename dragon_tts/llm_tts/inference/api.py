"""Public inference API for the Orpheus-style LLM-TTS (transformers backend).

Pipeline: reference clip -> speaker tokens, build prompt, LLM generates the
audio-token stream, codec decodes it back to a waveform.

Supports both SNAC (default) and NeuCodec codecs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import soundfile as sf
import torch
import torchaudio

from dragon_tts.llm_tts.codec_base import AudioCodec
from dragon_tts.llm_tts.sequence import build_prompt
from dragon_tts.llm_tts.vocab import END_AUDIO, parse_audio_token
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

AudioInput = Union[str, Path, torch.Tensor]

SPEAKER_SR = 16000


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


class OrpheusTTSPipeline:
    """Load the fine-tuned Qwen3 LLM + extended tokenizer + codec + speaker tok."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        speaker_ckpt: Union[str, Path],
        device: str = "cuda",
        codec: str = "snac",
        snac_model: Optional[str] = None,
        neucodec_model: Optional[str] = None,
        speaker_crop_seconds: float = 4.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = (
            AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.codec: AudioCodec = _build_codec(codec, device, snac_model, neucodec_model)
        self.speaker = SpeakerTokenizerPipeline(str(speaker_ckpt), device=device)
        self.speaker_crop = int(speaker_crop_seconds * SPEAKER_SR)
        self.end_audio_id = self.tokenizer.convert_tokens_to_ids(END_AUDIO)

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
        wav = self._load_ref_16k(ref).unsqueeze(0)  # (1, L)
        idx = self.speaker.tokenize(wav)  # (1, 1, 32)
        return idx[0, 0, :].cpu().tolist()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        ref_audio: AudioInput,
        max_new_tokens: int = 2000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Generate a waveform for ``text`` in the voice of ``ref_audio``."""
        spk_ids = self.speaker_tokens(ref_audio)
        prompt = build_prompt(spk_ids, text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.end_audio_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0, input_ids.shape[1] :].tolist()
        pairs = self._ids_to_audio_pairs(gen_ids)
        return self.codec.decode_pairs(pairs)

    def _ids_to_audio_pairs(self, gen_ids: List[int]) -> List[Tuple[int, int]]:
        tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)
        cb = self.codec.codebook_size
        nat = self.codec.num_audio_tokens
        pairs: List[Tuple[int, int]] = []
        for t in tokens:
            if t == END_AUDIO:
                break
            try:
                pairs.append(
                    parse_audio_token(t, codebook_size=cb, num_audio_tokens=nat)
                )
            except ValueError:
                continue  # skip stray non-audio tokens
        return pairs
