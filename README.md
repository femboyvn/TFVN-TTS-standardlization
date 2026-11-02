# teencode-normalizer

This normalizes Vietnamese **teencode** into standard Vietnamese. The model is a small encoder–decoder Transformer with about 19.9M parameters. It restores tones, expands abbreviations and works out pronouns from context. It is trained in **Python/PyTorch**, exported to **INT8 ONNX**, and run by a **Rust** binary on ONNX Runtime, which is small enough for a **Raspberry Pi 4**.

| input | output |
|---|---|
| `c oi, e nho c qua` | `chị ơi, em nhớ chị quá` |
| `c oi, c muon hoi cai nay` | `chú ơi, cháu muốn hỏi cái này` |
| `mua toi goi cc cao cap voi` | `mua tôi gói claude code cao cấp với` |
| `thang lam ML` | `thằng làm ML` |

> ⚠️ The model normalizes slang faithfully, **including vulgar slang**. This is intended behaviour. If you need to, filter the output downstream.

This repository is **code only**. The data and model live on Hugging Face:

| artifact | where | local path the code expects |
|---|---|---|
| dataset (train/dev/test, untoned, ctx, hard) | HF dataset `<hf-user>/teencode-vi-v3` | `data/v3/*.jsonl.gz` |
| model (INT8 ONNX + tokenizer.json, output_vocab.txt, config.json, guard.json) | HF model `<hf-user>/teencode-normalizer-v3`, folder `onnx/` | `models/teencode-normalizer-v3/onnx/` |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <hf-user>/teencode-vi-v3 --repo-type dataset --local-dir data/v3
huggingface-cli download <hf-user>/teencode-normalizer-v3 --local-dir models/teencode-normalizer-v3
```

## Layout
```
src/train/   Python: data prep + BPE tokenizer (step_1), trainer (step_2), ONNX export + INT8 (step_3),
             guard tuning / eval (step_4_eval*, eval_int8_test, eval_bf16, colab_eval), encode_data.py, train_v3.sh, requirements.txt
src/infer/   Rust crate: step_4_inference (beam 4 + guard, CLI, stdin, --serve HTTP)
docker/      Dockerfile + entrypoint.sh + compose: image that pulls the model from HF at start
.github/     workflows: docker.yml (multi-arch image -> ghcr.io), ci.yml (cargo build, py_compile)
colab/       Colab GPU training notebook
AGENTS.md    onboarding for contributors and AI agents
```

## Inference (Rust)
You need the ONNX Runtime **1.22.x** shared library (`onnxruntime-linux-x64-1.22.0.tgz` or `-aarch64-` from the ONNX Runtime GitHub releases).
```bash
cargo build --release --manifest-path src/infer/Cargo.toml
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.22.0
B=src/infer/target/release/step_4_inference
$B --model-dir models/teencode-normalizer-v3/onnx "c oi, e nho c qua"        # -> chị ơi, em nhớ chị quá
echo "thang lam ML" | $B --model-dir models/teencode-normalizer-v3/onnx
$B --model-dir models/teencode-normalizer-v3/onnx --serve 0.0.0.0:8080      # POST /normalize {"text"} or {"texts":[...]}, GET /health
```
Flags: `--beam N`, `--greedy`, `--no-guard`, `--threads N`, `--time`, `--fp32`.

## Docker (amd64 and arm64 / Raspberry Pi 4)
GitHub Actions (`.github/workflows/docker.yml`) builds the image and pushes it to GHCR. The model isn't baked in: the container downloads it from Hugging Face the first time it starts.
```bash
docker pull ghcr.io/<github-user>/teencode-normalizer:latest          # multi-arch, works on the Pi (arm64)
docker run -d -p 8080:8080 -v teencode-models:/models \
  -e HF_REPO=<hf-user>/teencode-normalizer-v3 ghcr.io/<github-user>/teencode-normalizer:latest
```
After the first workflow run, make the package public: GitHub → Packages → teencode-normalizer → Package settings → Change visibility. For settings, CLI mode, and a local build (`docker build -f docker/Dockerfile .`), see [docker/README.md](docker/README.md).

## Training (Python)
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r src/train/requirements.txt
# HF dataset -> trainer-ready dir with token ids (tokenizer from the HF model repo)
python src/train/encode_data.py --tokenizer models/teencode-normalizer-v3/onnx/tokenizer.json --in data/v3 --out data_v3
bash src/train/train_v3.sh                  # stages: data train export eval (STAGES="export eval" ...), DEVICE=cuda for GPU
```
- Raw pairs instead of the released data: `python src/train/step_1_prepare_data.py --out data_new gold.jsonl.gz --extra untoned=u.jsonl:0.5 --extra-test hard=hard.jsonl`. Add `--keep-tokenizer` to fine-tune with the existing vocab.
- Colab: `colab/teencode_train_colab.ipynb`.

The dataset is jsonl(.gz) with one `{"input", "output"[, "category", "w", "src_name"]}` per line. The dev/test files also carry `src`/`tgt` token ids. `encode_data.py` adds those ids wherever they are missing.

## Results (v3, CER, INT8 ONNX, beam 4 + guard)
| test set | v2 | **v3 INT8** | v3 BF16 |
|---|---|---|---|
| gold (999) | 2.92% | **2.08%** | 2.05% |
| untoned (986) | 12.38% | **11.19%** | 11.37% |
| context (803, in-distribution) | 11.93% | **0.35%** | 0.23% |
| hard set (1,251) | 10.96% | **4.95%** | 4.84% |

Rust INT8 takes about 0.07 s per sentence on an x86 CPU. On a Raspberry Pi 4 the estimate is 0.25–0.55 s per sentence, which hasn't been measured yet.

## Known limitations
- `cc` sometimes becomes "cái" instead of "claude code".
- `ml` depends on capitalization.
- "claude code" comes out lowercase.
- `b` next to `cháu` should become "bác".
- Fully untoned text is still at about 11% CER.

## License
MIT (code). The dataset and model licenses are on their Hugging Face cards.
