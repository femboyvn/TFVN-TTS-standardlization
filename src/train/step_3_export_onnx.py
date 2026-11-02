"""STEP 3 (Python companion) — export best checkpoint to ONNX and INT8-quantize for Raspberry Pi 4.

Invoked by `step_3_export_onnx.rs`. Produces in --out (default onnx/):
  encoder.onnx / decoder.onnx            FP32 graphs
  encoder.int8.onnx / decoder.int8.onnx  dynamic INT8 (onnxruntime.quantization.quantize_dynamic, QInt8 weights;
                                         dynamic quantization runs on ARM64 CPUs such as the Pi 4's Cortex-A72)
  tokenizer.json                         copied from data/ so the onnx/ folder is self-contained
Graph I/O (no KV cache: the decoder re-reads the whole prefix each step; cheap at ~20-token sentences):
  encoder: src_ids int64 [1,S]                                  -> memory float [1,S,256]
  decoder: tgt_ids int64 [1,T], memory [1,S,256], src_ids [1,S] -> logits float [1,8000] (last position)
Then verifies: greedy decode with PyTorch vs ONNX FP32 vs ONNX INT8 on test examples.
"""
import argparse, json, os, shutil, numpy as np, torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from step_2_train import Seq2Seq, BOS, EOS, PAD

class Enc(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, src_ids): return s.m.encode(src_ids)

class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, tgt_ids, memory, src_ids): return s.m.decode(tgt_ids, memory, src_ids)[:, -1]

def ort_greedy(enc, dec, src, max_len):
    src = np.array([src], dtype=np.int64); mem = enc.run(None, {"src_ids": src})[0]; ys = [BOS]
    for _ in range(max_len):
        nxt = int(dec.run(None, {"tgt_ids": np.array([ys], dtype=np.int64), "memory": mem, "src_ids": src})[0][0].argmax())
        if nxt == EOS: break
        ys.append(nxt)
    return ys[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="onnx"); ap.add_argument("--verify", type=int, default=30)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(2)
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    m = Seq2Seq(ck["cfg"]); m.load_state_dict(ck["model"]); m.eval()
    print(f"[step3] loaded {a.ckpt} (cfg {ck['cfg']}, step {ck.get('step')}, params {sum(p.numel() for p in m.parameters()):,})")
    S, T = torch.tensor([[1, 5, 6, 7, 2]]), torch.tensor([[1, 5, 6]])
    with torch.no_grad():
        mem = m.encode(S)
        # wrappers must be .eval(): the exporter restores the wrapper's prior mode (train) onto m afterwards
        torch.onnx.export(Enc(m).eval(), (S,), f"{a.out}/encoder.onnx", input_names=["src_ids"], output_names=["memory"],
                          dynamic_axes={"src_ids": {1: "S"}, "memory": {1: "S"}}, opset_version=17, dynamo=False)
        torch.onnx.export(Dec(m).eval(), (T, mem, S), f"{a.out}/decoder.onnx", input_names=["tgt_ids", "memory", "src_ids"],
                          output_names=["logits"], dynamic_axes={"tgt_ids": {1: "T"}, "memory": {1: "S"}, "src_ids": {1: "S"}},
                          opset_version=17, dynamo=False)
    m.eval()  # export wrappers restore their own (train) mode on m; force eval again
    for n in ("encoder", "decoder"):
        quantize_dynamic(f"{a.out}/{n}.onnx", f"{a.out}/{n}.int8.onnx", weight_type=QuantType.QInt8)
    shutil.copy(f"{a.data}/tokenizer.json", f"{a.out}/tokenizer.json")
    json.dump({**ck["cfg"], "params": sum(p.numel() for p in m.parameters()), "ckpt": a.ckpt, "step": ck.get("step")},
              open(f"{a.out}/config.json", "w"), indent=1)  # model size travels with the ONNX folder
    for f in sorted(os.listdir(a.out)): print(f"[step3] {a.out}/{f}: {os.path.getsize(f'{a.out}/{f}') / 1e6:.1f} MB")

    # ---- verification ---------------------------------------------------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.data}/tokenizer.json")
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = {k: ort.InferenceSession(f"{a.out}/{k}.onnx", so) for k in ("encoder", "decoder", "encoder.int8", "decoder.int8")}
    rows = [json.loads(l) for l in open(f"{a.data}/test.jsonl", encoding="utf-8")][: a.verify]
    if not rows: print("[step3] --verify 0: skipping verification"); return
    # logits check on one example
    r = rows[0]; s = torch.tensor([r["src"]]); t = torch.tensor([r["tgt"][:-1]])
    with torch.no_grad(): ref = m.decode(t, m.encode(s), s)[:, -1].numpy()
    mem32 = sess["encoder"].run(None, {"src_ids": s.numpy()})[0]
    got = sess["decoder"].run(None, {"tgt_ids": t.numpy(), "memory": mem32, "src_ids": s.numpy()})[0]
    print(f"[step3] FP32 ONNX vs PyTorch max |logit diff| = {np.abs(ref - got).max():.2e}")
    agree32 = agree8 = ok8 = 0
    for r in rows:
        with torch.no_grad(): pt = [x for x in m.greedy(torch.tensor([r["src"]]), max_len=len(r["src"]) + 24)[0].tolist()[1:] if x != PAD]
        pt = pt[:pt.index(EOS)] if EOS in pt else pt
        o32 = ort_greedy(sess["encoder"], sess["decoder"], r["src"], len(r["src"]) + 23)
        o8 = ort_greedy(sess["encoder.int8"], sess["decoder.int8"], r["src"], len(r["src"]) + 23)
        agree32 += o32 == pt; agree8 += o8 == pt; ok8 += tok.decode(o8) == r["output"]
    for r in rows[:3]:
        print(f"  in : {r['input']}\n  ref: {r['output']}\n  int8: {tok.decode(ort_greedy(sess['encoder.int8'], sess['decoder.int8'], r['src'], len(r['src']) + 23))}")
    n = len(rows)
    print(f"[step3] greedy output identical to PyTorch: FP32 {agree32}/{n}, INT8 {agree8}/{n}; INT8 exact-match vs reference {ok8}/{n}")

if __name__ == "__main__":
    main()
