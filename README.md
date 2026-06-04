# Dragon-TTS

Hệ thống TTS dạng modular. Phần hiện tại đã implement là **speaker tokenizer**: chuyển speaker embedding thành chuỗi token rời rạc ("global tokens") để các TTS model khác dùng làm speaker conditioning. Các component khác (semantic tokenizer, vocoder, text encoder...) sẽ sit alongside phần này khi được thêm vào.

Thiết kế speaker tokenizer lấy cảm hứng từ phần global-token của [SparkVox BiCodec](https://github.com/SparkAudio/SparkVox), nhưng tách rời hoàn toàn khỏi semantic encoder / audio reconstruction:

```
audio (24 kHz) → mel (128 mels) → [frozen ECAPA-TDNN] → x_vector (1024-d)
                                                     ↘ features (1536, T)
                                                       → PerceiverResampler → (32, 128)
                                                       → ResidualFSQ (levels=[4]*6) → indices (1, 32)
                                                       → project → d_vector (1024-d)

loss = MSE(d_vector, x_vector.detach())  [+ optional cosine loss]
```

Chỉ phần **PerceiverResampler + ResidualFSQ + projection** được train. ECAPA-TDNN load từ pretrained checkpoint và freeze.

## Cấu trúc thư mục

```
dragon_tts/
├── modules/                  # Building blocks dùng chung (vendor từ SparkVox).
│   ├── ecapa/                # ECAPA-TDNN
│   ├── fsq/                  # FSQ + ResidualFSQ
│   └── perceiver_encoder.py  # PerceiverResampler
└── speaker_tokenizer/        # Component speaker tokenizer
    ├── model.py              # `SpeakerTokenizer` nn.Module
    ├── data/                 # mel, audio_dataset, datamodule
    ├── training/lit_module.py
    └── inference/api.py
configs/speaker_tokenizer/    # Hydra configs
scripts/speaker_tokenizer/    # train.py, load_ecapa_ckpt.py, sanity_check.py
tests/speaker_tokenizer/      # round-trip unit tests
```

Khi thêm component mới (vocoder, semantic_tokenizer...), tạo sibling trong từng top-level tree (`dragon_tts/<comp>/`, `scripts/<comp>/`, `tests/<comp>/`, `configs/<comp>/`).

## Quick start

```bash
pip install -e .

# 1. Extract ECAPA weights từ checkpoint (BiCodec ckpt hoặc ECAPA standalone)
python scripts/speaker_tokenizer/load_ecapa_ckpt.py \
    --src <user_ckpt.pt> --dst ./ckpts/ecapa_tdnn.pt

# 2. Train
python scripts/speaker_tokenizer/train.py --config-name base \
    data.manifest=<train.jsonl> \
    ecapa_ckpt=./ckpts/ecapa_tdnn.pt

# 3. Sanity check
python scripts/speaker_tokenizer/sanity_check.py \
    ckpt=./outputs/.../best.ckpt manifest=<eval.jsonl>
```

Manifest format (JSONL, mỗi dòng):
```json
{"wav_path": "/path/to/audio.wav", "speaker_id": "spk_001"}
```
`speaker_id` chỉ dùng để chia val theo speaker, không bắt buộc.

## Inference

```python
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

pipe = SpeakerTokenizerPipeline(ckpt_path="path/to/best.ckpt", device="cuda")
indices = pipe.tokenize("audio.wav")          # (1, num_quantizers, token_num) — global tokens
d_vector = pipe.detokenize(indices)            # (1, 1024)
```

## Licensing

Apache-2.0. Một số module trong `dragon_tts/modules/{ecapa,fsq,perceiver_encoder}` được vendored từ SparkVox (Apache-2.0). Xem `NOTICE`.
