# Dragon-TTS

Hệ thống TTS dạng modular. Phần hiện tại đã implement là **speaker tokenizer**: chuyển speaker embedding thành chuỗi token rời rạc ("global tokens") để các TTS model khác dùng làm speaker conditioning. Các component khác (semantic tokenizer, vocoder, text encoder...) sẽ sit alongside phần này khi được thêm vào.

Thiết kế speaker tokenizer lấy cảm hứng từ phần global-token của [SparkVox BiCodec](https://github.com/SparkAudio/SparkVox), nhưng tách rời hoàn toàn khỏi semantic encoder / audio reconstruction:

```
                ┌─ [frozen ECAPA-TDNN]  (mel 128, 16 kHz)  → x_vector (1024-d), features (1536, T)
audio (16 kHz) ─┤
                └─ [frozen WavLM-base-plus-sv] (waveform)   → x_vector (512-d),  features (768, T)
                                                     ↓
                          features → PerceiverResampler → (32, 128)
                                   → ResidualFSQ (levels=[4]*6) → indices (1, 32)
                                   → project → d_vector (== x_vector dim)

loss = MSE(d_vector, x_vector.detach())  [+ optional cosine loss]
```

Chỉ phần **PerceiverResampler + ResidualFSQ + projection** được train. Speaker encoder (ECAPA-TDNN hoặc WavLM) luôn **frozen** và ở `eval()` mode.

### Lựa chọn speaker encoder

Encoder chọn qua config block `model.encoder.type`, áp dụng cho cả training lẫn inference (encoder được lưu trong checkpoint nên inference tự nhận đúng loại):

| `encoder.type` | Input            | x_vector / out_dim | context_dim | Trọng số                                   |
| -------------- | ---------------- | ------------------ | ----------- | ------------------------------------------ |
| `ecapa`        | mel 128 @ 16 kHz | 1024               | 1536        | `encoder.ckpt` (từ `load_ecapa_ckpt.py`)   |
| `wavlm`        | waveform @ 16 kHz| 512                | 768         | `from_pretrained` (`microsoft/wavlm-base-plus-sv`) |

`out_dim` và `context_dim` được suy ra tự động từ backbone — không cần chỉnh hyperparameter nào khác. Dataset tự emit mel hay waveform tuỳ `encoder.type`.

## Cấu trúc thư mục

```
dragon_tts/
├── modules/                  # Building blocks dùng chung (vendor từ SparkVox).
│   ├── ecapa/                # ECAPA-TDNN
│   ├── fsq/                  # FSQ + ResidualFSQ
│   └── perceiver_encoder.py  # PerceiverResampler
└── speaker_tokenizer/        # Component speaker tokenizer
    ├── model.py              # `SpeakerTokenizer` nn.Module
    ├── backbones.py          # SpeakerBackbone: EcapaBackbone / WavLMBackbone + build_backbone
    ├── data/                 # mel, audio_dataset, datamodule
    ├── training/lit_module.py
    └── inference/api.py
configs/speaker_tokenizer/    # Hydra configs: base.yaml (ecapa), wavlm.yaml
scripts/speaker_tokenizer/    # train.py, load_ecapa_ckpt.py, sanity_check.py
tests/speaker_tokenizer/      # round-trip unit tests (ecapa + wavlm)
```

Khi thêm component mới (vocoder, semantic_tokenizer...), tạo sibling trong từng top-level tree (`dragon_tts/<comp>/`, `scripts/<comp>/`, `tests/<comp>/`, `configs/<comp>/`).

## Quick start

```bash
pip install -e .
# WavLM encoder cần thêm `transformers` (>=4.40).

# ===== Option A: ECAPA-TDNN encoder (--config-name base) =====
# 1. Extract ECAPA weights từ checkpoint (BiCodec ckpt hoặc ECAPA standalone)
python scripts/speaker_tokenizer/load_ecapa_ckpt.py \
    --src <user_ckpt.pt> --dst ./ckpts/ecapa_tdnn.pt

# 2. Train
python scripts/speaker_tokenizer/train.py --config-name base \
    data.manifest=<train.jsonl> \
    model.encoder.ckpt=./ckpts/ecapa_tdnn.pt

# ===== Option B: WavLM-base-plus-sv encoder (--config-name wavlm) =====
# Không cần extract ckpt — trọng số nạp qua from_pretrained.
python scripts/speaker_tokenizer/train.py --config-name wavlm \
    data.manifest=<train.jsonl>

# 3. Sanity check (tự nhận encoder từ checkpoint)
python scripts/speaker_tokenizer/sanity_check.py \
    ckpt=./outputs/.../best.ckpt manifest=<eval.jsonl>
```

Manifest format (JSONL, mỗi dòng):
```json
{"wav_path": "/path/to/audio.wav", "speaker_id": "spk_001"}
```
`speaker_id` chỉ dùng để chia val theo speaker, không bắt buộc.

## Inference

Pipeline tự đọc loại encoder từ checkpoint và tiền xử lý audio đúng cách (mel cho ECAPA, waveform cho WavLM) — API giống nhau cho cả hai:

```python
from dragon_tts.speaker_tokenizer.inference.api import SpeakerTokenizerPipeline

pipe = SpeakerTokenizerPipeline(ckpt_path="path/to/best.ckpt", device="cuda")
indices = pipe.tokenize("audio.wav")          # (1, num_quantizers, token_num) — global tokens
d_vector = pipe.detokenize(indices)            # (1, out_dim): 1024 (ecapa) hoặc 512 (wavlm)
```

## LLM-TTS (Orpheus-style)

Component thứ hai: TTS tự hồi quy theo phong cách [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) — một LLM (Qwen3-0.6B) sinh chuỗi token âm thanh **SNAC** (`hubertsiuzdak/snac_24khz`, 7 token/frame), điều kiện hoá bằng **speaker token** từ speaker tokenizer ở trên. Định dạng chuỗi:

```
<|task_tts|><|start_embedding_token|>{speaker tokens}<|end_embedding_token|><|start_content|>{text}<|end_content|><|start_audio_token|>{SNAC audio tokens}<|end_audio_token|>
```

Cài thêm dependency: `pip install "snac>=1.2" "transformers>=4.51"` (và `vllm` nếu muốn streaming).

```bash
# 0. Tạo tokenizer mở rộng (+ tuỳ chọn resize model) — dùng cho cả train lẫn inference.
PYTHONPATH=. $PY scripts/llm_tts/build_tokenizer.py --config-name base \
    tokenizer.out_dir=./ckpts/llm_tts_qwen3 tokenizer.save_model=true

# 1. Sinh training data JSONL ({"text": <sequence>}) từ CSV LJSpeech.
PYTHONPATH=. $PY scripts/llm_tts/build_training_data.py --config-name base \
    data.csv_path=/path/LJSpeech-1.1/metadata.csv data.wavs_dir=/path/LJSpeech-1.1/wavs \
    speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt output=./data/llm_tts_train.jsonl

# 2. (Bạn tự train LLM trên JSONL ở trên.)

# 3. Inference — transformers, hoặc vLLM streaming (infer_vllm.py).
PYTHONPATH=. $PY scripts/llm_tts/infer.py --config-name base \
    model_dir=./ckpts/llm_tts_qwen3 speaker_ckpt=./ckpts/speaker_tokenizer/last.ckpt \
    infer.text="Xin chào thế giới" infer.ref_audio=/path/ref.wav infer.out=./out.wav
```

Manifest đầu vào là **CSV kiểu LJSpeech** (`id|transcription|normalized_transcription`); transcript lấy từ cột text. Cột/delimiter cấu hình trong `configs/llm_tts/base.yaml`.

## Licensing

Apache-2.0. Một số module trong `dragon_tts/modules/{ecapa,fsq,perceiver_encoder}` được vendored từ SparkVox (Apache-2.0). Xem `NOTICE`.
