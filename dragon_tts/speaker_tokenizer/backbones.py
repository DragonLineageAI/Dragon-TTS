"""Speaker-embedding backbones for the SpeakerTokenizer.

A backbone turns audio into:
  - ``x_vector`` ``(B, embed_dim)`` — the pooled speaker embedding that the
    tokenizer learns to reconstruct (the training target).
  - ``features`` ``(B, context_dim, T)`` — frame-level features that feed the
    PerceiverResampler as cross-attention context.

Four backbones are provided, selectable via config:
  - ``ecapa``    — frozen ECAPA-TDNN on LINEAR mel (the original design).
  - ``wavlm``    — frozen ``microsoft/wavlm-base-plus-sv`` on raw 16 kHz waveform.
  - ``redimnet`` — frozen ``IDRnD/ReDimNet`` on raw 16 kHz waveform.
  - ``qwen3``    — frozen Qwen3-TTS ECAPA-TDNN on log-mel @ 24 kHz waveform.

Everything downstream (PerceiverResampler → ResidualFSQ → projection) is
encoder-agnostic and only needs ``context_dim`` / ``embed_dim`` to be sized
correctly, plus ``input_kind`` so the data pipeline knows whether to feed mel
or waveform.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from dragon_tts.modules.ecapa.ecapa_tdnn import ECAPA_TDNN_GLOB_c512


class SpeakerBackbone(nn.Module):
    """Common interface for speaker-embedding backbones."""

    #: "mel" → expects mel-spectrogram (B, n_mels, T); "waveform" → raw (B, T).
    input_kind: str = "mel"

    @property
    def context_dim(self) -> int:
        """Channel dim ``C`` of ``features`` (PerceiverResampler dim_context)."""
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        """Dim ``E`` of ``x_vector`` (== tokenizer out_dim / projection output)."""
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_vector (B, E), features (B, C, T))``."""
        raise NotImplementedError


class EcapaBackbone(SpeakerBackbone):
    """Frozen ECAPA-TDNN on LINEAR mel features.

    Numerically identical to the original inline wiring: same module, same
    ``(B, n_mels, T) -> (B, T, n_mels)`` transpose before ECAPA, same
    ``return_latent=True`` call.
    """

    input_kind = "mel"

    def __init__(
        self,
        feat_dim: int = 128,
        embed_dim: int = 1024,
        channels: int = 512,
    ):
        super().__init__()
        assert channels == 512, (
            "Only ECAPA_TDNN_GLOB_c512 is vendored; context_dim is channels*3 "
            "= 1536. Adjust if you vendor another ECAPA variant."
        )
        self._channels = channels
        self._embed_dim = embed_dim
        self.encoder = ECAPA_TDNN_GLOB_c512(feat_dim=feat_dim, embed_dim=embed_dim)

    @property
    def context_dim(self) -> int:
        return self._channels * 3  # cat(out2, out3, out4) → 1536

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def load_encoder_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ):
        """Load a standalone ECAPA state_dict (bare keys) into the encoder."""
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def forward(
        self, mels: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = mels.transpose(1, 2)  # ECAPA expects (B, T, F)
        x_vector, features = self.encoder(x, True)  # (B, embed_dim), (B, 1536, T)
        return x_vector, features


class ReDimNetBackbone(SpeakerBackbone):
    """Frozen ReDimNet (IDRnD/ReDimNet) on raw 16 kHz waveform.

    Loads a pretrained ``ReDimNetWrap`` via ``torch.hub`` and splits its
    forward into two parts:

    1. ``spec → backbone(return_all_outputs=True)`` → frame-level 1D features
       ``(B, C*F, T)`` used as PerceiverResampler context.
    2. ``pool → bn → linear`` → 192-d pooled speaker embedding used as the
       reconstruction target (``x_vector``).

    Available model variants: b0–b6, M.  See
    https://github.com/IDRnD/ReDimNet/blob/master/EVALUATION.md
    """

    input_kind = "waveform"

    def __init__(
        self,
        model_name: str = "M",
        train_type: str = "ft_mix",
        dataset: str = "vb2+vox2+cnc",
        freeze: bool = True,
    ):
        super().__init__()
        self.model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
        )
        # Enable frame-level output from the backbone.
        self.model.backbone.return_all_outputs = True
        self.model.return_all_outputs = True

        # Derive dims from the loaded model config.
        self._context_dim = int(self.model.backbone.C * self.model.backbone.F)
        self._embed_dim = int(self.model.linear.out_features)  # 192
        self._freeze = freeze

    @property
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Step 1: Mel-spectrogram (internal to ReDimNet, NOT the ECAPA mel).
        spec = self.model.spec(wav)
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)

        # Step 2: Backbone → frame-level features.
        backbone_out, _all_outs = self.model.backbone(spec)
        features = backbone_out  # (B, C*F, T)

        # Step 3: Pool → BN → Linear → x_vector.
        x_vector = self.model.linear(self.model.bn(self.model.pool(backbone_out)))
        if self.model.bn2 is not None and not isinstance(
            self.model.bn2, torch.nn.Identity
        ):
            x_vector = self.model.bn2(x_vector)

        return x_vector, features


class WavLMBackbone(SpeakerBackbone):
    """Frozen ``WavLMForXVector`` (microsoft/wavlm-base-plus-sv) on raw waveform.

    The pooled xvector ``embeddings`` (512-d) is the reconstruction target; the
    base encoder's last hidden state (768-ch) is the PerceiverResampler context.
    Both come from a single forward (``output_hidden_states=True``).
    """

    input_kind = "waveform"

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus-sv",
        do_normalize: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        # Imported lazily so the rest of the package works without transformers.
        from transformers import WavLMForXVector

        self.model = WavLMForXVector.from_pretrained(pretrained)
        self.do_normalize = do_normalize
        self._context_dim = int(self.model.config.hidden_size)  # 768
        self._embed_dim = int(self.model.config.xvector_output_dim)  # 512
        # `freeze` here only records intent; the SpeakerTokenizer owns the actual
        # requires_grad / eval state via freeze_encoder(). Kept for clarity.
        self._freeze = freeze

    @property
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _normalize(self, wav: torch.Tensor) -> torch.Tensor:
        # Per-utterance zero-mean/unit-var, matching the SV feature extractor.
        # Done in fp32 with an epsilon guard for silent/padded crops.
        wav = wav.float()
        mean = wav.mean(dim=-1, keepdim=True)
        std = wav.std(dim=-1, keepdim=True)
        return (wav - mean) / (std + 1e-7)

    def forward(
        self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if self.do_normalize:
            wav = self._normalize(wav)
        out = self.model(input_values=wav, output_hidden_states=True)
        x_vector = out.embeddings  # (B, 512)
        features = out.hidden_states[-1].transpose(1, 2)  # (B, 768, T)
        return x_vector, features


class Qwen3EcapaBackbone(SpeakerBackbone):
    """Frozen Qwen3-TTS ECAPA-TDNN speaker encoder.

    Loads ``EcapaTdnnSpeakerEncoder`` via
    ``AutoModel.from_pretrained(pretrained, trust_remote_code=True)``.

    **Training path** (``input_kind="waveform_raw"``):
    The dataset emits full-length raw waveform (no crop/extend).
    ``Qwen3Collator`` (backed by ``EcapaTdnnFeatureExtractor``) converts
    waveforms to padded log-mel ``(B, T, 128)`` + ``attention_mask``
    before the backbone ever sees the data.  So ``forward()`` receives
    **pre-computed mel**, not raw waveform.

    **Inference path**:
    Use ``preprocess()`` to convert raw waveform(s) to mel + mask, then
    pass to ``forward()``.

    x_vector:  ``(B, 2048)``  — pooled speaker embedding (reconstruction target).
    features:  ``(B, 1536, T)`` — frame-level features after MFA (PerceiverResampler context).
    """

    input_kind = "waveform_raw"

    def __init__(
        self,
        pretrained: str = "./ckpts/Qwen3-Voice-Embedding-12Hz-1.7B",
        sample_rate: int = 24000,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(pretrained, trust_remote_code=True)
        self._sample_rate = sample_rate
        self._context_dim = int(self.model.config.enc_channels[-1])  # 1536
        self._embed_dim = int(self.model.config.enc_dim)  # 2048
        self._freeze = freeze

        # Keep processor for inference path (preprocess()).
        self._processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def preprocess(
        self,
        raw_speech,
        sampling_rate: int | None = None,
    ):
        """Convert raw waveform(s) to mel + attention_mask via the processor.

        Convenience method for inference (no DataLoader / collator).

        Args:
            raw_speech: ``np.ndarray``, ``list[np.ndarray]``, ``torch.Tensor``,
                or file path ``str``.
            sampling_rate: sample rate of the input audio(s).

        Returns:
            dict with ``input_values`` ``(B, T, 128)`` and
            ``attention_mask`` ``(B, T)`` tensors.
        """
        import numpy as np

        if isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.numpy()
        sr = sampling_rate or self._sample_rate
        features = self._processor(raw_speech, sampling_rate=sr)
        return {
            "input_values": features["input_values"],
            "attention_mask": features["attention_mask"],
        }

    def forward(
        self, mel: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on pre-computed log-mel.

        Args:
            mel: ``(B, T_frames, n_mels)`` log-mel spectrogram, already padded.
                Produced by ``Qwen3Collator`` (training) or ``preprocess()``
                (inference).
            attention_mask: ``(B, T_frames)`` long mask.  1 = real frame,
                0 = padding.  ``None`` = all frames valid.
        """
        out = self.model(
            input_values=mel,
            attention_mask=attention_mask,
            return_features=True,
        )
        x_vector = out.last_hidden_state  # (B, 2048)
        features = out.hidden_states[0]   # (B, 1536, T)
        return x_vector, features


def build_backbone(encoder_cfg: Dict[str, Any]) -> SpeakerBackbone:
    """Construct a backbone from a config dict with a ``type`` key.

    ``type="ecapa"``    → ``EcapaBackbone(feat_dim, embed_dim, channels)``
    ``type="wavlm"``    → ``WavLMBackbone(pretrained, do_normalize, freeze)``
    ``type="redimnet"`` → ``ReDimNetBackbone(model_name, train_type, dataset, freeze)``
    ``type="qwen3"``    → ``Qwen3EcapaBackbone(pretrained, sample_rate, freeze)``

    Extra keys not consumed by the backbone (e.g. ``ckpt``) are ignored here;
    the SpeakerTokenizer / LightningModule handle checkpoint loading.
    """
    cfg = dict(encoder_cfg)
    enc_type = cfg.pop("type")
    if enc_type == "ecapa":
        return EcapaBackbone(
            feat_dim=cfg.get("feat_dim", 128),
            embed_dim=cfg.get("embed_dim", 1024),
            channels=cfg.get("channels", 512),
        )
    if enc_type == "wavlm":
        return WavLMBackbone(
            pretrained=cfg.get("pretrained", "microsoft/wavlm-base-plus-sv"),
            do_normalize=cfg.get("do_normalize", True),
            freeze=cfg.get("freeze", True),
        )
    if enc_type == "redimnet":
        return ReDimNetBackbone(
            model_name=cfg.get("model_name", "M"),
            train_type=cfg.get("train_type", "ft_mix"),
            dataset=cfg.get("dataset", "vb2+vox2+cnc"),
            freeze=cfg.get("freeze", True),
        )
    if enc_type == "qwen3":
        return Qwen3EcapaBackbone(
            pretrained=cfg.get("pretrained", "./ckpts/Qwen3-Voice-Embedding-12Hz-1.7B"),
            sample_rate=cfg.get("sample_rate", 24000),
            freeze=cfg.get("freeze", True),
        )
    raise ValueError(
        f"Unknown encoder type: {enc_type!r} (expected 'ecapa', 'wavlm', 'redimnet', or 'qwen3')"
    )

