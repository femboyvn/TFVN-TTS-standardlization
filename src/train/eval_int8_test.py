"""Evaluate an exported INT8 ONNX model dir (encoder.int8/decoder.int8 + guard.json) on data/test.jsonl
with the same beam search + guard as step_4_eval.py. Usage: eval_int8_test.py <model_dir> [threads] [test.jsonl]"""
import sys, json, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
import step_2_train as S
from step_4_eval import guard, score, strip
d = sys.argv[1]; th = int(sys.argv[2]) if len(sys.argv) > 2 else 2
so = ort.SessionOptions(); so.intra_op_num_threads = th
enc = ort.InferenceSession(f"{d}/encoder.int8.onnx", so); dec = ort.InferenceSession(f"{d}/decoder.int8.onnx", so)
tok = Tokenizer.from_file(f"{d}/tokenizer.json"); g = json.load(open(f"{d}/guard.json"))
vocab = set(open(f"{d}/output_vocab.txt", encoding="utf-8").read().split("\n"))
al = ((g["align_max_in"], g["align_max_out"], True) if g.get("align_one_to_one") else (g["align_max_in"], g["align_max_out"])) if g.get("align_max_in") else None
rt = bool(g.get("retone", False))
tf = sys.argv[3] if len(sys.argv) > 3 else "data/test.jsonl"
def logsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
def beam_one(src_ids, beam=4, alpha=0.6):
    src = np.array([src_ids], dtype=np.int64); mem = enc.run(None, {"src_ids": src})[0]; max_len = min(192, len(src_ids) + 24)
    beams = [([S.BOS], 0.0)]; done = []
    for _ in range(max_len - 1):
        ys = np.array([b[0] for b in beams], dtype=np.int64); n = len(beams)
        lp = logsm(np.concatenate([dec.run(None, {"tgt_ids": ys[k:k+1], "memory": mem, "src_ids": src})[0].reshape(1, -1) for k in range(n)]).astype(np.float64))
        cand = []
        for (seq, sc), row in zip(beams, lp):
            idx = np.argsort(-row)[:beam]; cand += [(seq + [int(t)], sc + float(row[t])) for t in idx]
        cand.sort(key=lambda x: -x[1]); beams = []
        for seq, sc in cand:
            if seq[-1] == S.EOS: done.append((seq, sc / (((5 + len(seq)) / 6) ** alpha)))
            else: beams.append((seq, sc))
            if len(beams) == beam: break
        if len(done) >= beam or not beams: break
    if not done: done = [(s, sc / (((5 + len(s)) / 6) ** alpha)) for s, sc in beams]
    return strip(max(done, key=lambda x: x[1])[0])
test = S.load(tf)
hy = [tok.decode(beam_one(r["src"], g.get("beam", 4))) for r in test]
raw = score(test, hy); gd = score(test, [guard(r["input"], h, vocab, g["max_edit"], g["word_repair"], al, rt) for r, h in zip(test, hy)])
cp = score(test, [r["input"] for r in test])
print(json.dumps(dict(model_dir=d, test=tf, n=len(test), guard=g, raw_cer=raw[0], raw_em=raw[1], guarded_cer=gd[0], guarded_em=gd[1], copy_cer=cp[0], copy_em=cp[1])))
