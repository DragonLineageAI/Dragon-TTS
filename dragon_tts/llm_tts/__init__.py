"""Orpheus-style autoregressive LLM-TTS component.

Models speech as a discrete token stream that a causal LLM (Qwen3) learns to
generate, in the custom prompt format::

    <|task_tts|>
    <|start_embedding_token|>{speaker tokens}<|end_embedding_token|>
    <|start_content|>{text}<|end_content|>
    <|start_audio_token|>{audio tokens}<|end_audio_token|>

Speaker tokens come from the trained ``speaker_tokenizer`` component; audio
tokens come from the selected neural codec:

  - **SNAC** (``hubertsiuzdak/snac_24khz``): 7 tokens per frame, 4096 codebook,
    following the Orpheus layout.
  - **NeuCodec** (``neuphonic/neucodec``): 1 token per frame, 65536 codebook
    (FSQ), 50 Hz frame rate.

The codec is selected via config (``codec: snac`` or ``codec: neucodec``).
"""
