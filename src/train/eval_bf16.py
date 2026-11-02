"""BF16/FP32 PyTorch eval of a checkpoint with the same beam4 + guard.json as the INT8 eval.
Usage: eval_bf16.py --prec bf16|fp32 --ckpt ckpt/best.pt --guard-dir onnx --test FILE [--threads 2] [--out out.json]
       eval_bf16.py --prec bf16 --samples "s1" "s2" ..."""
import argparse, json, time, collections, contextlib, torch
from tokenizers import Tokenizer
import step_2_train as S
from step_4_eval import beam_one, guard
ap = argparse.ArgumentParser()
ap.add_argument("--prec", default="bf16"); ap.add_argument("--ckpt", default="ckpt/best.pt")
ap.add_argument("--guard-dir", default="onnx"); ap.add_argument("--test"); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out"); ap.add_argument("--samples", nargs="*")
a = ap.parse_args(); torch.set_num_threads(a.threads)
d = a.guard_dir; tok = Tokenizer.from_file(f"{d}/tokenizer.json"); g = json.load(open(f"{d}/guard.json"))
vocab = set(open(f"{d}/output_vocab.txt", encoding="utf-8").read().split("\n"))
al = ((g["align_max_in"], g["align_max_out"], True) if g.get("align_one_to_one") else (g["align_max_in"], g["align_max_out"])) if g.get("align_max_in") else None
rt = bool(g.get("retone", False))
ck = torch.load(a.ckpt, weights_only=False, map_location="cpu"); m = S.Seq2Seq(ck["cfg"]); m.load_state_dict(ck["model"]); m.eval()
ctx = (lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if a.prec == "bf16" else contextlib.nullcontext
def run(text, src=None):
    if src is None: src = [S.BOS] + tok.encode(text).ids + [S.EOS]
    with ctx(): h = tok.decode(beam_one(m, src, g.get("beam", 4), g.get("alpha", 0.6)))
    return h, guard(text, h, vocab, g["max_edit"], g["word_repair"], al, rt)
if a.samples:
    for s in a.samples: h, gh = run(s); print(json.dumps(dict(prec=a.prec, input=s, raw=h, guarded=gh), ensure_ascii=False))
    raise SystemExit
rows = [r for r in S.load(a.test) if r.get("input", "").strip()]
t0 = time.time(); outs = []
for r in rows: outs.append(run(r["input"].strip(), r.get("src")))
dt = time.time() - t0
def sc(k):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r, o in zip(rows, outs):
        for c in (r.get("category") or "uncategorized", "ALL"):
            x = agg[c]; ref = r["output"].strip(); x[0] += 1; x[1] += S.edit_distance(o[k], ref); x[2] += len(ref); x[3] += o[k] == ref
    return {c: dict(n=x[0], cer=x[1] / x[2], em=x[3] / x[0]) for c, x in agg.items()}
res = dict(prec=a.prec, test=a.test, n=len(rows), threads=a.threads, sec_per_sent=dt / len(rows), raw=sc(0), guarded=sc(1))
print(json.dumps(res))
if a.out: json.dump(dict(res, outputs=outs), open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
