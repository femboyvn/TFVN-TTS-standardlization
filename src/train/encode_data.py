"""Add token-id fields ("src"/"tgt" = [BOS] + BPE ids + [EOS]) to released jsonl(.gz) files so the Python
trainer/eval scripts can use them directly. Same ids as the v3 data prep (checked on 20k train rows).

  python src/train/encode_data.py --tokenizer models/teencode-normalizer-v3/onnx/tokenizer.json --in data/v3 --out data_v3
Every *.jsonl / *.jsonl.gz in --in becomes <out>/<name>.jsonl; tokenizer.json + output_vocab.txt are copied
when present in --in (else taken from the tokenizer's folder). Rows longer than --max-len tokens are dropped.
"""
import argparse, gzip, json, os, shutil
from tokenizers import Tokenizer

BOS, EOS = 1, 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="models/teencode-normalizer-v3/onnx/tokenizer.json")
    ap.add_argument("--in", dest="inp", default="data/v3"); ap.add_argument("--out", default="data_v3")
    ap.add_argument("--max-len", type=int, default=192)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    for extra in ("tokenizer.json", "output_vocab.txt"):
        src = os.path.join(a.inp, extra)
        if not os.path.exists(src): src = os.path.join(os.path.dirname(a.tokenizer), extra)
        if os.path.exists(src): shutil.copy(src, os.path.join(a.out, extra))
    for f in sorted(os.listdir(a.inp)):
        if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")): continue
        name = f[:-3] if f.endswith(".gz") else f
        op = gzip.open if f.endswith(".gz") else open
        kept = dropped = 0
        with op(os.path.join(a.inp, f), "rt", encoding="utf-8") as fi, open(os.path.join(a.out, name), "w", encoding="utf-8") as fo:
            for line in fi:
                r = json.loads(line)
                if "src" not in r or "tgt" not in r:
                    r["src"] = [BOS] + tok.encode(r["input"], add_special_tokens=False).ids + [EOS]
                    r["tgt"] = [BOS] + tok.encode(r["output"], add_special_tokens=False).ids + [EOS]
                if max(len(r["src"]), len(r["tgt"])) > a.max_len: dropped += 1; continue
                fo.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"); kept += 1
        print(f"[encode] {name}: {kept} rows ({dropped} dropped > {a.max_len} tokens)")

if __name__ == "__main__":
    main()
