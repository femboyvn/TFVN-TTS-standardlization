"""STEP 4 (eval companion) — per-category CER / exact match on the HARD test set (pronoun pairs, ambiguous
abbreviations, untoned context, user slang, ...).

Runs the real deployment path: the Rust `step_4_inference` binary on INT8 ONNX models (same beam + guard as the Pi),
so numbers match what users get. Rows need {"input","output"} and optionally "category" (missing -> "uncategorized").
Input can be the raw hard-set jsonl or data/test_hard.jsonl written by step 1 `--extra-test hard=PATH`.

Usage:
  ORT_DYLIB_PATH=... python src/train/step_4_eval_hard.py --test data/test_hard.jsonl \
      --models deployed=onnx new=onnx_v3 [--no-guard] [--beam 4] [--threads 4] [--samples 3] [--json out.json]
Prints one table per model: category, n, CER, EM, plus a "copy input" baseline and the overall row,
then the worst examples per category for the last model.
CER = char edit distance(hyp, ref) / len(ref), summed over rows of the category (same as steps 2/4 eval).
"""
import argparse, collections, json, os, subprocess, sys

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def run_model(bin_, model_dir, inputs, args):
    cmd = [bin_, "--model-dir", model_dir, "--threads", str(args.threads)]
    if args.beam: cmd += ["--beam", str(args.beam)]
    if args.no_guard: cmd += ["--no-guard"]
    # one line per input; newlines inside inputs would break the 1:1 mapping
    p = subprocess.run(cmd, input="\n".join(x.replace("\n", " ") for x in inputs) + "\n",
                       capture_output=True, text=True, check=True)
    out = p.stdout.rstrip("\n").split("\n")
    assert len(out) == len(inputs), f"{model_dir}: got {len(out)} outputs for {len(inputs)} inputs\n{p.stderr[-2000:]}"
    return out

def score(rows, hyps):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, errs, chars, em
    for r, h in zip(rows, hyps):
        for c in (r.get("category") or "uncategorized", "ALL"):
            a = agg[c]; a[0] += 1; a[1] += edit_distance(h, r["output"]); a[2] += len(r["output"]); a[3] += h == r["output"]
    return {c: dict(n=a[0], cer=a[1] / max(1, a[2]), em=a[3] / a[0]) for c, a in agg.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_hard.jsonl")
    ap.add_argument("--models", nargs="+", default=["deployed=onnx"], help="NAME=MODEL_DIR ...")
    ap.add_argument("--bin", default="src/infer/target/release/step_4_inference")
    ap.add_argument("--beam", type=int, default=0, help="0 = use the model dir's guard.json beam")
    ap.add_argument("--no-guard", action="store_true"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3, help="worst examples per category to print (last model)")
    ap.add_argument("--json", default=None, help="write all scores + outputs here")
    a = ap.parse_args()
    if not os.environ.get("ORT_DYLIB_PATH"): sys.exit("set ORT_DYLIB_PATH (see README)")
    rows = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("input", "").strip() and r.get("output", "").strip()]
    for r in rows: r["input"], r["output"] = r["input"].strip(), r["output"].strip()
    inputs = [r["input"] for r in rows]
    results = {"copy_input": (inputs, score(rows, inputs))}
    for spec in a.models:
        name, d = spec.split("=", 1); hyps = run_model(a.bin, d, inputs, a); results[name] = (hyps, score(rows, hyps))
    cats = sorted({c for _, s in results.values() for c in s if c != "ALL"}) + ["ALL"]
    names = list(results)
    print(f"hard test: {a.test} ({len(rows)} rows), beam {a.beam or 'guard.json'}, guard {'off' if a.no_guard else 'on'}")
    hdr = f"{'category':24s} {'n':>5s} " + " ".join(f"{n[:18]:>18s}" for n in names)
    print(hdr + "\n" + " " * 31 + " ".join(f"{'CER / EM':>18s}" for _ in names)); print("-" * len(hdr))
    for c in cats:
        n = results[names[0]][1][c]["n"]
        cells = " ".join(f"{results[k][1][c]['cer'] * 100:7.2f}% /{results[k][1][c]['em'] * 100:6.1f}%" for k in names)
        print(f"{c:24s} {n:5d} {cells}")
    last = names[-1]; hyps = results[last][0]
    print(f"\nworst examples per category ({last}):")
    by = collections.defaultdict(list)
    for r, h in zip(rows, hyps): by[r.get("category") or "uncategorized"].append((edit_distance(h, r["output"]) / max(1, len(r["output"])), r, h))
    for c in sorted(by):
        for e, r, h in sorted(by[c], key=lambda x: -x[0])[: a.samples]:
            if e == 0: continue
            print(f"  [{c}] in : {r['input']}\n  {' ' * len(c)}   ref: {r['output']}\n  {' ' * len(c)}   hyp: {h}")
    if a.json:
        json.dump({k: dict(scores=s, outputs=h) for k, (h, s) in results.items()}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
