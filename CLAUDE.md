# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Dragon-TTS is intended to be a multi-component TTS system. Currently only the **speaker tokenizer** component is implemented: it quantizes per-clip speaker identity into a small set of discrete "global tokens" so that downstream TTS models can use them as speaker conditioning. Future components (e.g. semantic tokenizer, vocoder, text encoder) are expected to sit alongside it.

The speaker tokenizer architecture mirrors SparkVox BiCodec's `SpeakerEncoder` (`/home/anhnct/project/SparkVox/sparkvox/models/codec/BiCodec/modules/speaker_encoder.py`) — frozen ECAPA-TDNN → PerceiverResampler (32 latents) → ResidualFSQ → linear projection — but is decoupled from any audio-reconstruction objective. The only training signal is reconstructing the frozen `x_vector` from the quantized `d_vector`.

## Repo layout

The repo is organized so that each component owns its own `dragon_tts/<component>/`, `scripts/<component>/`, `tests/<component>/`, and `configs/<component>/` subtree:

```
dragon_tts/
├── modules/                    # Reusable building blocks shared across components.
│   ├── ecapa/                  # Vendored ECAPA-TDNN (speaker encoder backbone).
│   ├── fsq/                    # Vendored FSQ + ResidualFSQ (quantizer).
│   └── perceiver_encoder.py    # Vendored PerceiverResampler.
└── speaker_tokenizer/          # The speaker quantizer component.
    ├── model.py                # `SpeakerTokenizer` nn.Module (the actual model).
    ├── data/                   # Mel transform, audio dataset, datamodule.
    ├── training/lit_module.py  # LightningModule + cosine-warmup scheduler.
    └── inference/api.py        # `SpeakerTokenizerPipeline` public API.
scripts/speaker_tokenizer/      # train.py, load_ecapa_ckpt.py, sanity_check.py.
configs/speaker_tokenizer/      # Hydra configs (base.yaml).
tests/speaker_tokenizer/        # Round-trip unit tests.
```

When adding a new component (e.g. a vocoder), create siblings under each of these top-level trees rather than nesting it inside `speaker_tokenizer/`. ECAPA/FSQ/PerceiverResampler should stay in `dragon_tts/modules/` so other components can reuse them.

## Environment

The user's conda env `sparktts` has torch, torchaudio, einops, einx, hydra, omegaconf, pytest. It is **missing** `pytorch_lightning`, which is required for training/inference (but **not** for the round-trip tests).

```bash
# Use the sparktts env explicitly:
PY=/home/anhnct/anaconda3/envs/sparktts/bin/python

# To train/run scripts, first install Lightning into that env:
$PY -m pip install "pytorch-lightning>=2.2"
```

Always run commands from the repo root and prepend `PYTHONPATH=.` so module paths resolve without an editable install.

## Common commands

```bash
# Run all speaker-tokenizer tests (only torch + einops + einx — no Lightning needed):
PYTHONPATH=. $PY -m pytest tests/speaker_tokenizer/test_round_trip.py -v

# Run a single test:
PYTHONPATH=. $PY -m pytest tests/speaker_tokenizer/test_round_trip.py::test_round_trip_deterministic -v

# 1) Extract ECAPA weights from user's checkpoint (BiCodec ckpt or ECAPA-only ckpt).
#    Auto-detects 3 prefix patterns: BiCodec-wrapped, `speaker_encoder.`-wrapped, or bare.
PYTHONPATH=. $PY scripts/speaker_tokenizer/load_ecapa_ckpt.py \
    --src <ckpt.pt> --dst ./ckpts/ecapa_tdnn.pt

# 2) Train (Hydra entry; config in configs/speaker_tokenizer/base.yaml):
PYTHONPATH=. $PY scripts/speaker_tokenizer/train.py --config-name base \
    data.manifest=<train.jsonl> ecapa_ckpt=./ckpts/ecapa_tdnn.pt

# Smoke run (a few batches, 2 epochs):
PYTHONPATH=. $PY scripts/speaker_tokenizer/train.py --config-name base \
    data.manifest=<jsonl> ecapa_ckpt=./ckpts/ecapa_tdnn.pt \
    trainer.max_epochs=2 trainer.limit_train_batches=20

# 3) Sanity check (round-trip cosine/MSE + codebook utilization + entropy):
PYTHONPATH=. $PY scripts/speaker_tokenizer/sanity_check.py \
    ckpt=outputs/speaker_tokenizer/ckpts/last.ckpt manifest=<eval.jsonl>
```

Manifest is JSONL, one object per line: `{"wav_path": "...", "speaker_id": "..."}`. `speaker_id` is optional and is used only to make the val split held-out by speaker.

## Architecture — what to know before editing

**Frozen-feature design.** ECAPA-TDNN is the **only** part that needs a pretrained checkpoint. It is always frozen and always in `eval()` mode (so BatchNorm running stats don't drift) — `SpeakerTokenizer.train(mode)` overrides the default `nn.Module` behavior to keep `self.speaker_encoder` in eval. Its forward is wrapped in `torch.no_grad()` when `_ecapa_frozen` is set. The only correct way to set this up is `model.freeze_ecapa()` or `model.load_ecapa_state_dict(sd)` — never set `requires_grad` by hand. Don't try to fine-tune ECAPA jointly with the quantizer using MSE only: without a discriminative loss, ECAPA collapses.

**Trainable params.** Only PerceiverResampler + ResidualFSQ projection layers + the final 1024×4096 Linear are trained. The FSQ codebook itself is **implicit** (a fixed grid in latent space), so there is no codebook-loss term — but also no commitment loss, which means **dead codes are silent**. Always trust the per-token-position codebook utilization metric logged by `lit_module.py` and surfaced explicitly in `scripts/speaker_tokenizer/sanity_check.py`.

**Loss.** `loss = mse_weight * F.mse_loss(d_vec, x_vec.detach()) + cosine_weight * (1 - F.cosine_similarity(d_vec, x_vec.detach())).mean()`. Default `cosine_weight = 0` (pure MSE matches the SparkVox `bicodec.py:88-89` convention). Bump it to 0.5–1.0 if sanity-check shows large EER gap vs raw ECAPA — no code changes needed, only the config.

**Indices shape gotcha.** Because `ResidualFSQ(is_channel_first=True)`, model output `indices` has shape `(B, num_quantizers, token_num)` — **not** `(B, token_num, num_quantizers)`. This matches SparkVox's `SpeakerEncoder` and is the convention `lit_module.py` and `sanity_check.py` index by. `detokenize` performs the necessary `transpose(1, 2)` internally; do not change this without auditing both callers.

**Mel config must match ECAPA training config.** `dragon_tts/speaker_tokenizer/data/mel.py` (`MelSpectrogramFeature`) reproduces the Spark-TTS-0.5B BiCodec settings (see `checkpoints/Spark-TTS-0.5B/BiCodec/config.yaml`): **16 kHz** / n_fft=1024 / win=640 / hop=320 / 128 mels / f_min=10 / slaney norm + slaney scale / **linear mel (power=1, NO log)**. The frozen ECAPA was trained on linear mel, so feeding log-mel or the wrong sample rate pushes ECAPA's frozen BatchNorms out of distribution and the `x_vector` magnitude explodes (norm ~1e6) → MSE in the 1e10 range and no learning. With the correct features `x_vector` norm is ~30 and MSE starts ~1.0. Verify against the source config when loading a new checkpoint.

**Vendored from SparkVox.** Five files under `dragon_tts/modules/{ecapa,fsq,perceiver_encoder.py}` are direct copies from SparkVox with Apache-2.0 headers preserved and imports relativized. See `NOTICE`. When updating vendored files, preserve the upstream behavior — including a latent bug in `residual_fsq.py:round_up_multiple` (references unimported `ceil`, never reached because `quantize_dropout_multiple_of=1` is the default and we never enable `quantize_dropout`).

**Default hyperparameters** (used by `SpeakerTokenizer` and `configs/speaker_tokenizer/base.yaml`): `input_dim=128`, `out_dim=1024`, `latent_dim=128`, `token_num=32`, `fsq_levels=[4]*6` (codebook_size = 4⁶ = 4096), `fsq_num_quantizers=1`, ECAPA channels=512 (hard-coded — context dim 1536 for PerceiverResampler depends on this).

## When debugging training

- `loss` not decreasing but `cos_sim_mean` already high at step 0 → expected: random PerceiverResampler can produce near-uniform `d_vec` ≈ `mean(x_vec)`, which has high cosine on detached targets. Look at codebook utilization and MSE absolute value, not just cosine.
- Codebook utilization < 5% at any position → FSQ collapse. Reduce `latent_dim`, change FSQ levels, or add cosine_weight to push the encoder to spread activations.
- ECAPA in train mode after `lit.train()` was called → bug. `SpeakerTokenizer.train()` keeps `speaker_encoder` in eval when `_ecapa_frozen=True`; verify `_ecapa_frozen` is True and don't bypass via `module.training` directly.
