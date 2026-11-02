"""STEP 4 (Python companion) — offline evaluation + tuning of decoding and the over-correction guard.

Same decoding/guard logic as `step_4_inference.rs`, used to TUNE on dev and REPORT on test:
  * decoding: greedy, or beam search (beam 4) with GNMT length penalty ((5+len)/6)^alpha
  * guard (post-processing, all optional, tuned on DEV only):
      - retone (for untoned input): alignment keys are diacritic-stripped, so re-toned words count as matches
      - one_to_one: equal-length mismatched spans (word-by-word substitutions) are accepted
      - LCS word alignment guard (main fix for over-correction): keep input words everywhere except small
        local rewrites (<= max_in input words -> 1..max_out output words); insertions/deletions/long rewrites rejected
      - word repair: if input and output have the same word count, any output word that is NOT in the
        training-output vocabulary (data/output_vocab.txt) is replaced by the input word at that position
      - sentence fallback: if char edit distance(input, output) / len(input) > --max-edit, or the output/input
        char length ratio is outside [0.6, 1.8], return the input unchanged
Writes the chosen settings to <out>/guard.json (read by step_4_inference.rs) and data/output_vocab.txt.

Usage: python src/train/step_4_eval.py --ckpt ckpt_ft/best.pt [--beam] [--threads 2] [--out onnx]
"""
import argparse, json, os, collections, torch
from tokenizers import Tokenizer
import step_2_train as S

def ed(a, b): return S.edit_distance(a, b)

def strip(ids):
    ids = [t for t in ids if t not in (S.PAD, S.BOS)]
    return ids[:ids.index(S.EOS)] if S.EOS in ids else ids

@torch.no_grad()
def greedy_all(m, rows):
    out = []
    for k in range(0, len(rows), 64):
        ch = rows[k:k + 64]; src, _ = S.collate(ch, range(len(ch)))
        out += [strip(x) for x in m.greedy(src, max_len=min(192, src.shape[1] + 24)).tolist()]
    return out

@torch.no_grad()
def beam_one(m, src_ids, beam=4, alpha=0.6):
    src = torch.tensor([src_ids]); mem = m.encode(src); max_len = min(192, len(src_ids) + 24)
    beams = [([S.BOS], 0.0)]; done = []
    for _ in range(max_len - 1):
        ys = torch.tensor([b[0] for b in beams])
        lp = torch.log_softmax(m.decode(ys, mem.expand(len(beams), -1, -1), src.expand(len(beams), -1))[:, -1].float(), -1)
        cand = []
        for (seq, sc), row in zip(beams, lp):
            v, i = row.topk(beam)
            cand += [(seq + [int(t)], sc + float(s)) for s, t in zip(v, i)]
        cand.sort(key=lambda x: -x[1]); beams = []
        for seq, sc in cand:
            if seq[-1] == S.EOS: done.append((seq, sc / (((5 + len(seq)) / 6) ** alpha)))
            else: beams.append((seq, sc))
            if len(beams) == beam: break
        if len(done) >= beam or not beams: break
    if not done: done = [(s, sc / (((5 + len(s)) / 6) ** alpha)) for s, sc in beams]
    return strip(max(done, key=lambda x: x[1])[0])

import unicodedata
def strip_diacritics(w):
    """Remove Vietnamese tone/vowel marks (NFD + drop combining marks, đ->d). Mirrored in step_4_inference.rs."""
    w = unicodedata.normalize("NFD", w).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in w if not unicodedata.combining(c))

def lcs_segments(a, b):
    """Word-level LCS alignment -> list of (i1, i2, j1, j2, equal). Mirrored exactly in step_4_inference.rs."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    segs, i, j, si, sj = [], 0, 0, 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            if (si, sj) != (i, j): segs.append((si, i, sj, j, False))
            segs.append((i, i + 1, j, j + 1, True)); i += 1; j += 1; si, sj = i, j
        elif dp[i + 1][j] >= dp[i][j + 1]: i += 1
        else: j += 1
    if (si, sj) != (n, m): segs.append((si, n, sj, m, False))
    return segs

def align_guard(inp, out, max_in, max_out, retone=False, one_to_one=False):
    """Keep the INPUT except where the model made a small local rewrite: a non-matching span of <= max_in
    input words replaced by 1..max_out output words. Pure insertions/deletions and long rewrites
    (hallucinations, repeated words, dropped words) are rejected -> input words kept."""
    a, b = inp.split(), out.split(); res = []
    # retone: align on diacritic-stripped keys, so re-toning an untoned word ("khong" -> "không") counts as a
    # match and the output word is kept, however long the re-toned span is. Other rules unchanged.
    ka, kb = ([strip_diacritics(w) for w in a], [strip_diacritics(w) for w in b]) if retone else (a, b)
    for i1, i2, j1, j2, eq in lcs_segments(ka, kb):
        # one_to_one: a mismatched span with the same number of words on both sides (word-by-word substitutions,
        # e.g. "e ko bik" -> "em không biết") is accepted whatever its length
        if eq or (0 < i2 - i1 <= max_in and 0 < j2 - j1 <= max_out) or (one_to_one and i2 - i1 == j2 - j1): res += b[j1:j2]
        else: res += a[i1:i2]
    return " ".join(res)

def guard(inp, out, vocab, max_edit, repair, align=None, retone=False):
    if align: out = align_guard(inp, out, align[0], align[1], retone, len(align) > 2 and align[2])
    if repair:
        iw, ow = inp.split(), out.split()
        if len(iw) == len(ow):
            out = " ".join(o if (o in vocab or o == i or (retone and strip_diacritics(o) == strip_diacritics(i))) else i
                           for i, o in zip(iw, ow))
    if max_edit is not None:
        r = len(out) / max(1, len(inp))
        if ed(inp, out) / max(1, len(inp)) > max_edit or not (0.6 <= r <= 1.8): return inp
    return out

def score(rows, hyps, per_set=False):
    """CER/EM; with several sets (rows carry _set), returns the weight-averaged CER/EM (weights in global W)."""
    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r, h in zip(rows, hyps):
        b = by[r.get("_set", "gold")]; b[0] += ed(h, r["output"]); b[1] += len(r["output"]); b[2] += h == r["output"]; b[3] += 1
    res = {k: (v[0] / v[1], v[2] / v[3]) for k, v in by.items()}
    if per_set: return res
    ws = {k: W.get(k, 1.0) for k in res}; t = sum(ws.values())
    return sum(res[k][0] * ws[k] for k in res) / t, sum(res[k][1] * ws[k] for k in res) / t

def fmt(rows, hyps):
    return " | ".join(f"{k} CER {c*100:.2f}% EM {e*100:.1f}%" for k, (c, e) in score(rows, hyps, True).items())

W = {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_ft/best.pt"); ap.add_argument("--data", default="data")
    ap.add_argument("--beam", action="store_true"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--sets", default="", help="comma list NAME:WEIGHT of extra eval sets, e.g. untoned:0.5 "
                    "(uses <data>/dev_NAME.jsonl and test_NAME.jsonl); gold dev/test get weight 1-sum")
    ap.add_argument("--retone-grid", default="0,1", help="retone options to try (0/1)")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{a.data}/tokenizer.json")
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu"); m = S.Seq2Seq(ck["cfg"]); m.load_state_dict(ck["model"]); m.eval()
    # training-output vocabulary (words seen in train `output`)
    vocab = set()
    for l in open(f"{a.data}/train.jsonl", encoding="utf-8"): vocab.update(json.loads(l)["output"].split())
    with open(f"{a.data}/output_vocab.txt", "w", encoding="utf-8") as f: f.write("\n".join(sorted(vocab)))
    extra = [(x.split(":")[0], float(x.split(":")[1])) for x in a.sets.split(",") if x]
    sets = [("gold", f"{a.data}/dev.jsonl", f"{a.data}/test.jsonl", 1 - sum(w for _, w in extra))]
    sets += [(n, f"{a.data}/dev_{n}.jsonl", f"{a.data}/test_{n}.jsonl", w) for n, w in extra]
    dev = [dict(r, _set=n) for n, d, _, _ in sets for r in S.load(d)]
    test = [dict(r, _set=n) for n, _, t, _ in sets for r in S.load(t)]
    W.update({n: w for n, _, _, w in sets})
    print(f"[eval] ckpt {a.ckpt} step {ck['step']} | output vocab {len(vocab)} words")
    print(f"[eval] copy-input baseline: DEV {fmt(dev, [r['input'] for r in dev])} || TEST {fmt(test, [r['input'] for r in test])}")
    modes = {"greedy": lambda rows: [tok.decode(x) for x in greedy_all(m, rows)]}
    if a.beam: modes["beam4"] = lambda rows: [tok.decode(beam_one(m, r["src"])) for r in rows]
    best = None
    for name, fn in modes.items():
        hd = fn(dev)
        print(f"  dev {name} raw: {fmt(dev, hd)}")
        for rt in [bool(int(x)) for x in a.retone_grid.split(",")]:
         for al in (None, (1, 4), (2, 4), (3, 4), (1, 4, True), (2, 4, True)):
          for repair in (False, True):
            for me in (None, 0.3, 0.5):
                gh = [guard(r["input"], h, vocab, me, repair, al, rt) for r, h in zip(dev, hd)]
                cer, em = score(dev, gh)
                print(f"  dev {name:6s} retone={int(rt)} align={al} repair={int(repair)} max_edit={me}: wCER {cer*100:.2f}% wEM {em*100:.1f}% [{fmt(dev, gh)}]")
                if best is None or cer < best[0]: best = (cer, name, repair, me, al, rt)
    cer, name, repair, me, al, rt = best
    print(f"[eval] best on DEV: {name} retone={rt} align={al} repair={repair} max_edit={me} wCER {cer*100:.2f}%")
    ht = modes[name](test)
    gt = [guard(r["input"], h, vocab, me, repair, al, rt) for r, h in zip(test, ht)]
    print(f"[eval] TEST {name} raw: {fmt(test, ht)}")
    print(f"[eval] TEST {name} guarded: {fmt(test, gt)}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(decode=name, beam=4 if name == "beam4" else 1, alpha=0.6, word_repair=repair, retone=rt, max_edit=me, align_max_in=al[0] if al else 0, align_max_out=al[1] if al else 0, align_one_to_one=bool(al and len(al) > 2 and al[2]),
                   min_len_ratio=0.6, max_len_ratio=1.8), open(f"{a.out}/guard.json", "w"), indent=1)
    import shutil; shutil.copy(f"{a.data}/output_vocab.txt", f"{a.out}/output_vocab.txt")

if __name__ == "__main__":
    main()
