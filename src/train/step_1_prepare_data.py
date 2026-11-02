"""STEP 1: prepare data. This is a Python port of the original Rust step_1_prepare_data.rs.

* Loads labeled .jsonl / .jsonl.gz files (rows: {"input": teencode, "output": standard}). Removes duplicates and
  empty rows, shuffles with seed 42 and splits 98/1/1 into train/dev/test.
* Trains a BPE tokenizer: 8000 tokens shared by source and target, NFC normalizer, Metaspace pre-tokenizer.
  Special ids are fixed: <pad>=0 <s>=1 </s>=2 <unk>=3. With --keep-tokenizer it reuses <out>/tokenizer.json,
  so new feedback data can be trained on without changing the vocabulary.
* Encodes each pair as [BOS] + ids + [EOS] and drops pairs longer than 192 tokens. Writes
  <out>/{train,dev,test}.jsonl with {"input","output","src","tgt"[,"w","src_name","category"]}.

Options:
  --add-identity R / --add-partial-noise R   anti-over-correction augmentation (train only); v3 used 0.15 / 0.10
  --extra NAME=PATH[:W]   extra source: gets its own 1%/1% dev_NAME/test_NAME, and its train rows get loss weight W.
                          Rows that overlap gold or held-out text are dropped (leakage guard).
  --extra-test NAME=PATH  eval-only set -> test_NAME.jsonl; matching train rows are removed
  --exclude-ids FILE, --drop-low-confidence, --eval-labeler L   filters for extra sources
Note: the shuffle uses Python's RNG, so the splits are NOT byte-identical to the released v3 splits that the Rust
version made. To reproduce v3 exactly, use the original v3 data dir.

  python src/train/step_1_prepare_data.py --out data_v3 gold.jsonl.gz --add-identity 0.15 --add-partial-noise 0.10 \
      --extra untoned=untoned.jsonl:0.5 --extra ctx=ctx.jsonl:1.0 --extra-test hard_1251=hard.jsonl
"""
import argparse, collections, gzip, json, os, random
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

VOCAB_SIZE, MAX_LEN, SEED = 8000, 192, 42
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


def read_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            try: v = json.loads(line)
            except json.JSONDecodeError: continue
            a, b = v.get("input"), v.get("output")
            if not isinstance(a, str) or not isinstance(b, str): continue
            a, b = a.strip(), b.strip()
            if a and b: yield a, b, v


def augment(train, id_frac, noise_frac, rng):
    out = list(train)
    if id_frac + noise_frac <= 0: return out
    assert id_frac + noise_frac < 0.9, "augmentation fractions too large"
    total = len(train) / (1 - id_frac - noise_frac)
    n_id, n_noise = int(total * id_frac), int(total * noise_frac)
    pool = [b for a, b in train if a != b]; rng.shuffle(pool)
    out += [(pool[k % len(pool)],) * 2 for k in range(n_id)]
    subs = collections.defaultdict(collections.Counter)
    for a, b in train:
        aw, bw = a.split(), b.split()
        if len(aw) == len(bw):
            for x, y in zip(aw, bw):
                if x != y: subs[y][x] += 1
    table = {k: sorted((t, c) for t, c in v.items() if c >= 5) for k, v in subs.items()}
    table = {k: v for k, v in table.items() if v}
    made = tries = 0
    while made < n_noise and tries < n_noise * 20:
        tries += 1
        s = pool[rng.randrange(len(pool))]; words = s.split()
        cand = [i for i, w in enumerate(words) if w in table]
        if not cand: continue
        k = 1 + min(int(rng.random() < 0.35), len(cand) - 1)
        for i in rng.sample(cand, k):
            opts = table[words[i]]
            words[i] = rng.choices([t for t, _ in opts], weights=[c for _, c in opts])[0]
        out.append((" ".join(words), s)); made += 1
    print(f"[step1] augmentation: +{n_id} identity, +{made} partial-noise (subst table {len(table)} words) -> train {len(out)}")
    rng.shuffle(out)
    return out


def train_tokenizer(train, path):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tr = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, show_progress=False, special_tokens=SPECIALS)
    tk.train_from_iterator((s for ab in train for s in ab), trainer=tr)
    tk.save(path, pretty=True)
    return tk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="data")
    ap.add_argument("--keep-tokenizer", action="store_true")
    ap.add_argument("--add-identity", type=float, default=0.0); ap.add_argument("--add-partial-noise", type=float, default=0.0)
    ap.add_argument("--extra", action="append", default=[]); ap.add_argument("--extra-test", action="append", default=[])
    ap.add_argument("--exclude-ids", action="append", default=[])
    ap.add_argument("--drop-low-confidence", action="store_true"); ap.add_argument("--eval-labeler", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    cats, eval_sets, held_out = {}, [], set()
    for spec in a.extra_test:
        name, path = spec.split("=", 1); rows, seen = [], set()
        for x, y, v in read_rows(path):
            if (x, y) in seen: continue
            seen.add((x, y)); rows.append((x, y)); held_out |= {x, y}
            if isinstance(v.get("category"), str): cats[(x, y)] = v["category"]
        print(f"[step1] extra-test '{name}' ({path}): {len(rows)} eval-only rows"); eval_sets.append((name, rows))

    seen, pairs = set(), []
    for f in a.files:
        for x, y, _ in read_rows(f):
            if (x, y) not in seen: seen.add((x, y)); pairs.append((x, y))
    print(f"[step1] loaded {len(pairs)} unique pairs from {a.files}")

    rng = random.Random(SEED); rng.shuffle(pairs)
    n_dev = max(1, len(pairs) // 100)
    dev, test, train = pairs[:n_dev], pairs[n_dev:2 * n_dev], pairs[2 * n_dev:]
    kept = [p for p in train if p[0] not in held_out and p[1] not in held_out]
    if len(kept) != len(train): print(f"[step1] removed {len(train) - len(kept)} gold train rows overlapping --extra-test sets")
    train = augment(kept, a.add_identity, a.add_partial_noise, rng)

    tok_path = os.path.join(a.out, "tokenizer.json")
    if a.keep_tokenizer and os.path.exists(tok_path):
        print(f"[step1] --keep-tokenizer: reusing {tok_path}"); tok = Tokenizer.from_file(tok_path)
    else:
        tok = train_tokenizer(train, tok_path)
    print(f"[step1] vocab size = {tok.get_vocab_size()}")
    for i, s in enumerate(SPECIALS): assert tok.token_to_id(s) == i, f"special {s} must have id {i}"

    enc = lambda s: [1] + tok.encode(s, add_special_tokens=False).ids + [2]

    def write(fh, split, label, extra=None):
        kept = dropped = tot = 0
        for x, y in split:
            s, t = enc(x), enc(y)
            if len(s) > MAX_LEN or len(t) > MAX_LEN: dropped += 1; continue
            tot += len(s); j = {"input": x, "output": y, "src": s, "tgt": t}
            if extra: j["w"], j["src_name"] = extra[1], extra[0]
            if (x, y) in cats: j["category"] = cats[(x, y)]
            fh.write(json.dumps(j, ensure_ascii=False) + "\n"); kept += 1
        print(f"[step1] {label}: kept {kept}, dropped {dropped} (> {MAX_LEN} tok), avg src len {tot / max(1, kept):.1f}")

    opn = lambda n: open(os.path.join(a.out, n), "w", encoding="utf-8")
    train_w = opn("train.jsonl"); write(train_w, train, "train (gold)")
    for name, split in (("dev", dev), ("test", test)):
        with opn(f"{name}.jsonl") as fh: write(fh, split, name)

    excluded = set()
    for f in a.exclude_ids:
        for line in open(f, encoding="utf-8"):
            try: i = json.loads(line).get("id")
            except (json.JSONDecodeError, AttributeError): i = None
            i = i or line.strip()
            if i: excluded.add(i)
    gold_text = {s for p in pairs for s in p}
    for spec in a.extra:
        name, rest = spec.split("=", 1); path, wt = rest, 1.0
        if ":" in rest:
            p, w = rest.rsplit(":", 1)
            try: path, wt = p, float(w)
            except ValueError: pass
        n_ex = n_low = n_leak = n_dup = 0; seen_x, pref, other = set(), [], []
        for x, y, v in read_rows(path):
            if v.get("id") in excluded: n_ex += 1; continue
            if a.drop_low_confidence and v.get("low_confidence") is True: n_low += 1; continue
            rt = (v.get("reference_toned") or "").strip()
            if x in gold_text or y in gold_text or (rt and rt in gold_text) or x in held_out or y in held_out: n_leak += 1; continue
            if (x, y) in seen_x: n_dup += 1; continue
            seen_x.add((x, y))
            if isinstance(v.get("category"), str): cats[(x, y)] = v["category"]
            (pref if a.eval_labeler and v.get("labeler") == a.eval_labeler else other).append((x, y))
        total = len(pref) + len(other); nd = max(1, total // 100)
        print(f"[step1] extra '{name}' ({path}, weight {wt}): {total} rows kept; removed {n_ex} excluded-ids, "
              f"{n_low} low-confidence, {n_leak} gold-overlap, {n_dup} dups; {len(pref)} preferred-labeler rows")
        xr = random.Random(SEED); xr.shuffle(pref); xr.shuffle(other); pool = pref + other
        write(train_w, pool[2 * nd:], f"train ({name}, w={wt})", (name, wt))
        with opn(f"dev_{name}.jsonl") as fh: write(fh, pool[:nd], f"dev_{name}")
        with opn(f"test_{name}.jsonl") as fh: write(fh, pool[nd:2 * nd], f"test_{name}")
    train_w.close()
    for name, rows in eval_sets:
        with opn(f"test_{name}.jsonl") as fh: write(fh, rows, f"test_{name} (eval-only)")


if __name__ == "__main__":
    main()
