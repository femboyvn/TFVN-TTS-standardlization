"""Quick greedy dev eval (CER / exact match) of a checkpoint with PyTorch, GPU if available.
Usage: python colab_eval.py --ckpt runs/X/best.pt --data data [--n 1000]"""
import argparse, json, torch
from tokenizers import Tokenizer
from step_2_train import Seq2Seq, collate, edit_distance, load, PAD, EOS
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default="data")
ap.add_argument("--sets", default="dev,dev_untoned,dev_ctx"); ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = Seq2Seq(ck["cfg"]); m.load_state_dict(ck["model"]); m.to(dev).eval(); tok = Tokenizer.from_file(f"{a.data}/tokenizer.json")
res = {}
for name in a.sets.split(","):
    rows = load(f"{a.data}/{name}.jsonl")[: a.n]; errs = chars = em = 0
    with torch.no_grad():
        for k in range(0, len(rows), 64):
            ch = rows[k:k + 64]; src, _ = collate(ch, range(len(ch)))
            out = m.greedy(src.to(dev), max_len=min(ck["cfg"]["max_len"], src.shape[1] + 24))
            for r, ids in zip(ch, out.tolist()):
                ids = [t for t in ids[1:] if t != PAD]; ids = ids[:ids.index(EOS)] if EOS in ids else ids
                h = tok.decode(ids); errs += edit_distance(h, r["output"]); chars += len(r["output"]); em += h == r["output"]
    res[name] = dict(n=len(rows), cer=round(100 * errs / chars, 2), em=round(100 * em / len(rows), 1))
    print(f"{name}: n={len(rows)} greedy CER {res[name]['cer']}% EM {res[name]['em']}%", flush=True)
print(json.dumps({"ckpt": a.ckpt, "step": ck.get("step"), **res}))
