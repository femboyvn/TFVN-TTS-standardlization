# AGENTS.md — onboarding for AI coding agents and new contributors

This repo is a Vietnamese **teencode → standard Vietnamese** seq2seq normalizer. It uses a small Transformer that is trained in PyTorch, exported to INT8 ONNX and run by a **Rust** binary, with a Raspberry Pi 4 as the target. Read this whole file before changing code. Inference is Rust only (`src/infer`); Python (`src/train`) is for data, training, export and eval.

## Repo map (code only: data + model are on Hugging Face)
```
src/train/step_1_prepare_data.py  raw pairs -> dedup, seeded 98/1/1 split, 8k BPE (tokenizers), encode, --extra/--extra-test/--keep-tokenizer
src/train/encode_data.py          HF dataset jsonl(.gz) -> trainer-ready dir with src/tgt ids
src/train/step_2_train.py         model + trainer (CPU/CUDA, bf16 autocast, --init-from growth, multi dev sets)
src/train/step_3_export_onnx.py   ONNX export + quantize_dynamic INT8 + parity check
src/train/step_4_eval.py          guard/decoding TUNING on dev -> guard.json (Python guard mirrors the Rust guard)
src/train/eval_int8_test.py       INT8 beam+guard CER/EM on one test file
src/train/step_4_eval_hard.py     per-category hard-set eval through the Rust binary
src/train/eval_bf16.py            PyTorch BF16/FP32 reference eval + latency
src/train/colab_eval.py           dev eval used by the Colab notebook
src/train/train_v3.sh             the whole v3 pipeline, staged (data/train/export/eval)
src/train/requirements.txt
src/infer/                        Rust crate, bin step_4_inference: ONNX Runtime (ort 2.0.0-rc.10, load-dynamic), beam 4 + guard, CLI/stdin/--serve
docker/                           the only image: pulls the model from HF_REPO at start (Dockerfile, entrypoint.sh, compose, .env.example)
.github/workflows/docker.yml      builds the amd64 + arm64 image and pushes it to ghcr.io/<owner>/<repo> (latest on main, semver on v* tags, sha); PRs only build
.github/workflows/ci.yml          cargo build (src/infer) + clippy (advisory), py_compile (src/train), shell syntax checks
colab/                            GPU training notebook
```
Not in git (see .gitignore): `data/`, `data_*/`, `models/`, `ckpt*/`, `onnx*/`, `*.onnx`, `*.pt`, `*.safetensors`, `*.jsonl(.gz)`, `.env`.

## Setup
```bash
pip install -r src/train/requirements.txt "huggingface_hub[cli]"
huggingface-cli download <hf-user>/teencode-vi-v3 --repo-type dataset --local-dir data/v3      # dataset -> data/v3/*.jsonl.gz
huggingface-cli download <hf-user>/teencode-normalizer-v3 --local-dir models/teencode-normalizer-v3   # model -> models/teencode-normalizer-v3/onnx/
cargo build --release --manifest-path src/infer/Cargo.toml
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.22.0       # Windows: onnxruntime.dll
```
The runtime model dir has to contain `encoder.int8.onnx`, `decoder.int8.onnx`, `tokenizer.json`, `output_vocab.txt`, `config.json` and `guard.json`. Those are model artifacts from the HF model repo, not repo files.

## Run
```bash
src/infer/target/release/step_4_inference --model-dir models/teencode-normalizer-v3/onnx "c oi, e nho c qua"   # -> chị ơi, em nhớ chị quá
src/infer/target/release/step_4_inference --model-dir models/teencode-normalizer-v3/onnx --serve 0.0.0.0:8080
docker pull ghcr.io/<github-user>/teencode-normalizer:latest      # built by GitHub Actions, amd64 + arm64
docker run -d -p 8080:8080 -v teencode-models:/models -e HF_REPO=<hf-user>/teencode-normalizer-v3 ghcr.io/<github-user>/teencode-normalizer:latest
docker build -f docker/Dockerfile -t teencode .                     # local build, only for testing Dockerfile changes
```

## Train, export, evaluate
- Everything at once (after the HF downloads above): `bash src/train/train_v3.sh` (stages: `STAGES="data train export eval"`).
- Config H: 8 enc / 4 dec layers, d_model 256, 8 heads, d_ff 1536, 8k shared BPE, max_len 192, 19.94M params. The released v3 is step 13000. It was warm-started from v2 (`INIT=ckpt_v2/best.pt`, identity map, 1000 frozen steps, 5 epochs, cosine, lr 3e-4).
- GPU: `DEVICE=cuda bash src/train/train_v3.sh`. Colab: `colab/teencode_train_colab.ipynb`. CPU works too, just slowly.
- Export and INT8: `python src/train/step_3_export_onnx.py --ckpt CKPT --data DATA --out OUT`. After that, **re-tune the guard** with `step_4_eval.py --beam --sets untoned:0.3,ctx:0.3`.
- Current v3 numbers (CER, INT8, beam 4 + guard). **Don't regress these:**

  | gold 999 | untoned 986 | ctx 803 | hard 1251 | hard: abbrev / pronoun / untoned |
  |---|---|---|---|---|
  | 2.08% | 11.19% | 0.35% (in-distribution) | **4.95%** | 8.09% / 2.34% / 6.20% |

  BF16 PyTorch is about the same (2.05 / 11.37 / 0.23 / 4.84). Latency for Rust INT8 on x86 is about 0.07 s per sentence. The Pi 4 estimate of 0.25–0.55 s has not been measured.

## Retraining with new feedback pairs
1. Write the feedback as `{"input","output"}` jsonl. Add a `category` field when it fits.
2. Copy the current `tokenizer.json` into the out dir, then run `python src/train/step_1_prepare_data.py --keep-tokenizer ... --extra feedback=feedback.jsonl:1.0`. Keep the tokenizer, or the model has to be retrained from scratch.
3. Fine-tune: `--init-from ckpt_v3/best.pt --lr 2e-4 --epochs 1-2`, using the same `--dev-sets`.
4. Export, re-tune the guard, then evaluate on **test_hard_1251** and on gold/untoned/ctx against `models/teencode-normalizer-v3/onnx`. Ship only if hard CER improves and nothing else gets worse by more than 0.2 pts.
5. The maintainer uploads the new model to the HF model repo. Then update the results tables in the README and in this file.

## Conventions
- **Rust-first and practical.** Inference/runtime code goes in Rust (`src/infer`). Python (`src/train`) is only for data, training, export and eval. Keep the `step_N_*` file names. `src/` holds exactly two folders: `train/` and `infer/`.
- The Python guard (`step_4_eval.py`) and the Rust guard (`step_4_inference.rs`) **must stay in sync**. Change both, or neither.
- Domain rules:
  - `cc` = **Claude Code** (not "cái"/"con cặc") in tech contexts.
  - `c ơi, e …` → **chị / em**.
  - `c ơi, c …` → **chú / cháu**.
  - `co oi, c …` → cô / cháu.
  - Uppercase `ML` = machine learning.
- Vulgar slang gets normalized as-is. That is intended and must not be filtered in the model.
- Deliverables for the maintainer (Windows/PowerShell) must be under 25 MB per file. Split bigger files with `split -b 23M`, add SHA256SUMS, and rejoin with `cmd /c "copy /b a + b out"`.

## Known weaknesses (good first tasks: add targeted data)
- `ml` is ambiguous: "Thang ml kia" → mặt lồn (fine), but "thang Lam ml" → "thằng Lâm mặt lồn" (wrong, ML was meant). The model leans too much on capitalization. Needs pairs like "lam ml", "hoc ml".
- `cc` sometimes becomes "cái".
- "claude code" comes out lowercase instead of "Claude Code".
- `b` next to `cháu` should become **bác**, but often comes out as "bạn".
- Untoned input is still at ~11% CER.

## Don't
- Don't bake the model into the image, and don't add a second Dockerfile. GitHub Actions builds and publishes the image. The model always comes from HF at runtime.
- Don't push, publish or upload to Hugging Face on the maintainer's behalf. The maintainer does those.
- Don't commit data, models, checkpoints, ONNX files, zips, `.env` or tokens. This repo is code only. The data goes to the HF dataset `<hf-user>/teencode-vi-v3` and the model to `<hf-user>/teencode-normalizer-v3`.
- Don't change the tokenizer or `PAD/BOS/EOS/UNK = 0/1/2/3` for a fine-tune.
- Don't report ctx-test CER as a headline number, because it is in-distribution. Use hard_1251.
- Don't hardcode absolute paths. Use CLI args relative to the repo root.
- Don't tune the guard on test sets. Tune on dev only.
