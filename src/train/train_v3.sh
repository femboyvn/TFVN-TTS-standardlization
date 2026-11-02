#!/usr/bin/env bash
# Full v3 pipeline (config H: 8 enc / 4 dec, d_model 256, 8 heads, d_ff 1536, 8k BPE, 19.94M params).
# Run from the repo root. Model artifacts (tokenizer/config) from HF:
#   huggingface-cli download <hf-user>/teencode-normalizer-v3 --local-dir models/teencode-normalizer-v3
# Env overrides: DATA, CKPT, OUT, INIT (warm-start ckpt), EPOCHS, DEVICE, THREADS.
#   bash src/train/train_v3.sh               # everything: data -> train -> export/INT8 -> guard tune -> eval
#   STAGES="export eval" bash src/train/train_v3.sh
# GPU: DEVICE=cuda (bf16 autocast is automatic). Colab: see colab/teencode_train_colab.ipynb.
set -euo pipefail
DATA=${DATA:-data_v3}; CKPT=${CKPT:-ckpt_v3}; OUT=${OUT:-onnx_v3}; EPOCHS=${EPOCHS:-5}
DEVICE=${DEVICE:-auto}; THREADS=${THREADS:-6}; INIT=${INIT:-}
STAGES=${STAGES:-"data train export eval"}
PY=${PY:-python}
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has data; then
  # 1) data: in-repo eval splits + release-asset train split -> token ids
  # dataset from Hugging Face:  huggingface-cli download <hf-user>/teencode-vi-v3 --repo-type dataset --local-dir data/v3
  [ -f data/v3/train.jsonl.gz ] || { echo "data/v3/train.jsonl.gz missing: download the HF dataset <hf-user>/teencode-vi-v3 into data/v3"; exit 1; }
  $PY src/train/encode_data.py --tokenizer models/teencode-normalizer-v3/onnx/tokenizer.json --in data/v3 --out "$DATA"
  # (Re-building data from raw labelled jsonl instead: python src/train/step_1_prepare_data.py --help)
fi

if has train; then
  # 2) train. The released v3 was warm-started from the v2 checkpoint (4 enc, d_ff 1024):
  #    INIT=ckpt_v2/best.pt -> --init-from ... --init-map identity --freeze-init-steps 1000.
  #    Without INIT it trains config H from scratch (use more epochs, e.g. EPOCHS=12, and --lr 7e-4 --warmup 1500).
  ARGS=(--data "$DATA" --ckpt-dir "$CKPT" --epochs "$EPOCHS" --device "$DEVICE" --threads "$THREADS"
        --d-model 256 --heads 8 --d-ff 1536 --enc-layers 8 --dec-layers 4
        --sched cosine --eval-every 1000 --log "$CKPT.log"
        --dev-sets "$DATA/dev.jsonl:0.4,$DATA/dev_untoned.jsonl:0.3,$DATA/dev_ctx.jsonl:0.3")
  if [ -n "$INIT" ]; then ARGS+=(--init-from "$INIT" --init-map identity --freeze-init-steps 1000 --lr 3e-4 --warmup 500)
  else ARGS+=(--lr 7e-4 --warmup 1500); fi
  $PY src/train/step_2_train.py "${ARGS[@]}"
fi

if has export; then
  # 3) ONNX export (fp32) + dynamic INT8 quantization + parity check
  $PY src/train/step_3_export_onnx.py --ckpt "$CKPT/best.pt" --data "$DATA" --out "$OUT"
  # 4a) tune decoding + guard on DEV, report TEST, write $OUT/guard.json and $DATA/output_vocab.txt
  $PY src/train/step_4_eval.py --ckpt "$CKPT/best.pt" --data "$DATA" --beam --out "$OUT" --sets untoned:0.3,ctx:0.3
  cp "$DATA/output_vocab.txt" "$DATA/tokenizer.json" "$OUT/"
  [ -f "$OUT/config.json" ] || cp models/teencode-normalizer-v3/onnx/config.json "$OUT/"
fi

if has eval; then
  # 4b) INT8 beam+guard on every test split (same numbers as the README table)
  for t in test test_untoned test_ctx; do
    (cd scripts && $PY eval_int8_test.py "../$OUT" "$THREADS" "../$DATA/$t.jsonl")
  done
  # 4c) hard set, per category, through the Rust runner
  cargo build --release --manifest-path src/infer/Cargo.toml
  $PY src/train/step_4_eval_hard.py --test "$DATA/test_hard_1251.jsonl" --models "new=$OUT" "v3=models/teencode-normalizer-v3/onnx" \
      --bin src/infer/target/release/step_4_inference --json "$OUT/hard1251.json"
  # optional: BF16 / FP32 PyTorch reference: python src/train/eval_bf16.py --prec bf16 --ckpt $CKPT/best.pt --guard-dir $OUT --test $DATA/test.jsonl
fi
