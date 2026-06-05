"""Orpheus-style autoregressive LLM-TTS component.

Models speech as a discrete token stream that a causal LLM (Qwen3) learns to
generate, in the custom prompt format::

    <|task_tts|>
    <|start_embedding_token|>{speaker tokens}<|end_embedding_token|>
    <|start_content|>{text}<|end_content|>
    <|start_audio_token|>{SNAC audio tokens}<|end_audio_token|>

Speaker tokens come from the trained ``speaker_tokenizer`` component; audio
tokens come from the SNAC neural codec (``hubertsiuzdak/snac_24khz``) following
the Orpheus 7-tokens-per-frame layout.
"""
