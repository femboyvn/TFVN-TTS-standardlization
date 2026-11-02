"""STEP 2 (Python companion) — train the teencode -> standard Vietnamese seq2seq Transformer.

Run directly (python src/train/step_2_train.py ...).

Model (defaults match the original spec; size is configurable with --d-model --heads --d-ff --enc-layers
  --dec-layers --dropout, stored in every checkpoint's "cfg" so steps 3/4 and --resume need no flags):
  encoder-decoder Transformer, 4 enc + 4 dec layers, d_model 256, 8 heads, d_ff 1024, dropout 0.1,
  max length 192 tokens, shared 8k BPE vocab (from step 1), separate src/tgt embeddings + output projection.
  Attention is written with plain matmul/softmax so it exports cleanly to ONNX and quantizes well (step 3).
Training: AdamW + warmup/inverse-sqrt LR, label smoothing 0.1, token-bucketed batches,
  BF16 mixed precision via torch.autocast (CPU bf16; CUDA bf16 if a GPU is present).
Checkpoints (in --ckpt-dir):
  last.pt  — full state (model, optimizer, scheduler, step, epoch, best) for --resume
  best.pt  — best model by dev CER (ties broken by dev loss)
Durability: after every eval, last.pt, best.pt, the tokenizer and --log (train.log) are copied atomically to
  --backup-dir DIR: optional extra copy of checkpoints (default: disabled).
Multiple dev sets: --dev-sets "data/dev.jsonl:0.5,data/dev_untoned.jsonl:0.5" evaluates each (loss/CER/EM logged per
  set); best.pt is chosen by the weighted mean CER (ties: weighted dev loss).
Per-row loss weights: rows with a "w" field (e.g. silver extra sources from step 1 --extra NAME=PATH:0.5) are weighted
  in the token loss; gold rows default to 1.0.
Continued training on new data: re-run step 1 with --keep-tokenizer, then
  step_2_train --init-from ckpt/best.pt --lr 2e-4 --epochs N   (fresh optimizer, same tokenizer/weights).
Growing the model from a smaller checkpoint: --init-from ckpt_small/best.pt --enc-layers 6 [--init-map bottom|spread]
  copies every shape-matching tensor (embeddings, matching layers, norms, output); see init_partial().
"""
import argparse, json, math, os, random, shutil, time
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3
CFG = dict(vocab=8000, d_model=256, heads=8, d_ff=1024, enc_layers=4, dec_layers=4, dropout=0.1, max_len=192)


# ----------------------------------------------------------------------------- model
class MHA(nn.Module):
    """Multi-head attention with explicit ops (ONNX/INT8 friendly)."""
    def __init__(s, d, h, p):
        super().__init__(); s.h, s.dk = h, d // h
        s.q, s.k, s.v, s.o = (nn.Linear(d, d) for _ in range(4)); s.drop = nn.Dropout(p)
    def forward(s, x, mem, bias):  # bias: additive mask broadcastable to [B,h,Tq,Tk]
        B, Tq, _ = x.shape; Tk = mem.shape[1]
        q = s.q(x).view(B, Tq, s.h, s.dk).transpose(1, 2)
        k = s.k(mem).view(B, Tk, s.h, s.dk).transpose(1, 2)
        v = s.v(mem).view(B, Tk, s.h, s.dk).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(s.dk) + bias
        a = s.drop(torch.softmax(a.float(), -1).to(v.dtype))
        return s.o((a @ v).transpose(1, 2).reshape(B, Tq, -1))

class FFN(nn.Sequential):
    def __init__(s, d, f, p): super().__init__(nn.Linear(d, f), nn.ReLU(), nn.Dropout(p), nn.Linear(f, d))

class EncLayer(nn.Module):  # pre-LayerNorm for stable training
    def __init__(s, c):
        super().__init__(); d = c["d_model"]
        s.sa, s.ff = MHA(d, c["heads"], c["dropout"]), FFN(d, c["d_ff"], c["dropout"])
        s.n1, s.n2, s.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(c["dropout"])
    def forward(s, x, bias):
        h = s.n1(x); x = x + s.drop(s.sa(h, h, bias))
        return x + s.drop(s.ff(s.n2(x)))

class DecLayer(nn.Module):
    def __init__(s, c):
        super().__init__(); d = c["d_model"]
        s.sa, s.ca = MHA(d, c["heads"], c["dropout"]), MHA(d, c["heads"], c["dropout"])
        s.ff = FFN(d, c["d_ff"], c["dropout"])
        s.n1, s.n2, s.n3, s.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(c["dropout"])
    def forward(s, y, mem, self_bias, cross_bias):
        h = s.n1(y); y = y + s.drop(s.sa(h, h, self_bias))
        y = y + s.drop(s.ca(s.n2(y), mem, cross_bias))
        return y + s.drop(s.ff(s.n3(y)))

class Seq2Seq(nn.Module):
    def __init__(s, c=CFG):
        super().__init__(); s.c = c; d = c["d_model"]
        s.src_emb, s.tgt_emb = nn.Embedding(c["vocab"], d, padding_idx=PAD), nn.Embedding(c["vocab"], d, padding_idx=PAD)
        s.src_pos, s.tgt_pos = nn.Embedding(c["max_len"], d), nn.Embedding(c["max_len"], d)
        s.enc = nn.ModuleList(EncLayer(c) for _ in range(c["enc_layers"]))
        s.dec = nn.ModuleList(DecLayer(c) for _ in range(c["dec_layers"]))
        s.enc_norm, s.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        s.out = nn.Linear(d, c["vocab"]); s.drop = nn.Dropout(c["dropout"]); s.scale = math.sqrt(d)

    @staticmethod
    def pad_bias(ids):  # [B,T] -> [B,1,1,T] additive mask (-1e4 on PAD keys)
        return (ids == PAD).to(torch.float32)[:, None, None, :] * -1e4

    def encode(s, src):
        pos = torch.arange(src.shape[1], device=src.device)
        x = s.drop(s.src_emb(src) * s.scale + s.src_pos(pos))
        bias = s.pad_bias(src)
        for l in s.enc: x = l(x, bias)
        return s.enc_norm(x)

    def decode(s, tgt, mem, src):
        T = tgt.shape[1]; pos = torch.arange(T, device=tgt.device)
        y = s.drop(s.tgt_emb(tgt) * s.scale + s.tgt_pos(pos))
        causal = torch.triu(torch.full((T, T), -1e4, device=tgt.device), 1)[None, None]
        self_bias = causal + s.pad_bias(tgt); cross_bias = s.pad_bias(src)
        for l in s.dec: y = l(y, mem, self_bias, cross_bias)
        return s.out(s.dec_norm(y))

    def forward(s, src, tgt_in): return s.decode(tgt_in, s.encode(src), src)

    @torch.no_grad()
    def greedy(s, src, max_len=None):
        """Batched greedy decode (no KV cache — same algorithm as ONNX inference in step 4)."""
        max_len = max_len or s.c["max_len"]; mem = s.encode(src); B = src.shape[0]
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            nxt = s.decode(ys, mem, src)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == EOS
            if done.all(): break
        return ys


def init_partial(model, ck, mode="identity"):
    """Initialize `model` from checkpoint `ck` (a smaller/equal model), copying every tensor that fits.
    * Same config -> exact load.
    * Wider d_ff -> old FFN weights go into the first d_ff_old units; the new units' second-layer weights are zeroed,
      so the FFN computes exactly the old function at step 0 (Net2Net-style function-preserving growth).
    * More layers -> layer mapping by `mode`:
        identity (default): new layer i <- old layer i; extra top layers get random weights but ZERO output
                            projections (attention .o and ffn[3]), so with pre-LN they are exact identities at step 0
                            and the grown model reproduces the old model's outputs before any training.
        bottom:  like identity but extra layers stay fully random (not function-preserving).
        spread:  new layer i <- old layer round(i*(old-1)/(new-1)) (duplicates layers; not function-preserving).
    * Different d_model / heads -> those tensors cannot be reused and stay random.
    Returns the set of state-dict keys that were (fully or partially) copied from `ck`."""
    src, dst = ck["model"], model.state_dict()
    old = ck.get("cfg", CFG); new = model.c
    def mapl(i, n_old, n_new):
        if mode == "spread" and n_new > 1: return round(i * (n_old - 1) / (n_new - 1))
        return i if i < n_old else None
    copied, fresh_layers = set(), set()
    for k, v in dst.items():
        sk = k
        for stack, nk in (("enc", "enc_layers"), ("dec", "dec_layers")):
            if k.startswith(stack + "."):
                i = int(k.split(".")[1]); j = mapl(i, old.get(nk, 4), new[nk])
                if j is None: fresh_layers.add(f"{stack}.{i}.")
                sk = None if j is None else f"{stack}.{j}." + k.split(".", 2)[2]
        if sk not in src: continue
        sv = src[sk]
        if sv.shape == v.shape: dst[k] = sv.clone(); copied.add(k)
        elif ".ff." in k and sv.dim() == v.dim() and all(a <= b for a, b in zip(sv.shape, v.shape)):
            t = v.clone()
            if k.endswith("ff.3.weight"): t.zero_()          # new hidden units contribute nothing at step 0
            t[tuple(slice(0, n) for n in sv.shape)] = sv; dst[k] = t; copied.add(k)
    if mode == "identity":
        for k in dst:
            if any(k.startswith(p) for p in fresh_layers) and (".o." in k or ".ff.3." in k): dst[k].zero_()
    model.load_state_dict(dst)
    return copied


# ----------------------------------------------------------------------------- data
def load(path):
    with open(path, encoding="utf-8") as f: return [json.loads(l) for l in f]

BATCH_CAP = 0  # set from --batch-size

def batches(rows, max_tokens, shuffle, rng):
    """Bucket by length so batches have ~max_tokens (src+tgt) tokens each."""
    idx = sorted(range(len(rows)), key=lambda i: (len(rows[i]["src"]) + len(rows[i]["tgt"]), rng.random() if shuffle else 0))
    out, cur, cur_max = [], [], 0
    for i in idx:
        L = max(len(rows[i]["src"]), len(rows[i]["tgt"]))
        if cur and (max(cur_max, L) * (len(cur) + 1) * 2 > max_tokens or (BATCH_CAP and len(cur) >= BATCH_CAP)): out.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, L)
    if cur: out.append(cur)
    if shuffle: rng.shuffle(out)
    return out

def collate(rows, ids):
    def pad(seqs):
        T = max(map(len, seqs)); return torch.tensor([x + [PAD] * (T - len(x)) for x in seqs])
    return pad([rows[i]["src"] for i in ids]), pad([rows[i]["tgt"] for i in ids])


# ----------------------------------------------------------------------------- metrics
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

class _BatchDS(torch.utils.data.Dataset):
    """Wraps a precomputed list of batches so a DataLoader can collate them in worker processes."""
    def __init__(self, rows, order): self.rows, self.order = rows, order
    def __len__(self): return len(self.order)
    def __getitem__(self, k):
        b = self.order[k]; src, tgt = collate(self.rows, b)
        return src, tgt, torch.tensor([float(self.rows[i].get("w", 1.0)) for i in b])

def evaluate(model, dev, tok, args, amp):
    device = next(model.parameters()).device
    model.eval(); tot_loss = tot_tok = 0
    with torch.no_grad(), amp():
        for b in batches(dev, args.max_tokens, False, random.Random(0)):
            src, tgt = collate(dev, b); src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            tot_loss += F.cross_entropy(logits.float().transpose(1, 2), tgt[:, 1:], ignore_index=PAD, reduction="sum").item()
            tot_tok += (tgt[:, 1:] != PAD).sum().item()
        # CER / exact match on a fixed subset (greedy decode is the slow part)
        sub = dev[: args.cer_samples]; errs = chars = em = 0
        for k in range(0, len(sub), 64):
            chunk = sub[k:k + 64]; src, _ = collate(chunk, range(len(chunk))); src = src.to(device)
            out = model.greedy(src, max_len=min(CFG["max_len"], src.shape[1] + 24))
            for r, ids in zip(chunk, out.tolist()):
                ids = [t for t in ids[1:] if t not in (PAD,)]; ids = ids[:ids.index(EOS)] if EOS in ids else ids
                hyp = tok.decode(ids); errs += edit_distance(hyp, r["output"]); chars += len(r["output"]); em += hyp == r["output"]
    model.train()
    return tot_loss / tot_tok, errs / max(chars, 1), em / max(len(sub), 1)


def backup(args):
    """Copy last.pt / best.pt / tokenizer / train.log to --backup-dir (atomic tmp+rename)."""
    if not args.backup_dir: return
    try:
        os.makedirs(args.backup_dir, exist_ok=True)
        for src in (f"{args.ckpt_dir}/last.pt", f"{args.ckpt_dir}/best.pt", f"{args.data}/tokenizer.json", args.log):
            if src and os.path.exists(src):
                dst = os.path.join(args.backup_dir, os.path.basename(src))
                shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    except OSError as e:
        print(f"[backup] WARNING: {e}", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--epochs", type=float, default=12); ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=7e-4); ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--cer-samples", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="resume from ckpt-dir/last.pt (model+optim+step)")
    ap.add_argument("--init-from", default=None, help="fine-tune: load weights only (e.g. ckpt/best.pt), fresh optimizer")
    ap.add_argument("--no-bf16", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backup-dir", default="", help="optional dir to mirror last.pt/best.pt into (\"\" = off)")
    ap.add_argument("--log", default="train.log", help="log file copied into --backup-dir")
    ap.add_argument("--sched", choices=["invsqrt", "cosine"], default="invsqrt",
                    help="invsqrt: warmup + inverse-sqrt (from-scratch default); cosine: warmup + cosine decay to "
                         "--min-lr-ratio*lr over the whole run (recommended for --init-from fine-tuning)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    # ---- model size (ignored on --resume: the checkpoint's cfg wins) ----
    ap.add_argument("--d-model", type=int, default=None, help="default 256 (or the --init-from ckpt's value)")
    ap.add_argument("--heads", type=int, default=None, help="default 8")
    ap.add_argument("--d-ff", type=int, default=None, help="default 1024")
    ap.add_argument("--enc-layers", type=int, default=None, help="default 4")
    ap.add_argument("--dec-layers", type=int, default=None, help="default 4")
    ap.add_argument("--dropout", type=float, default=None, help="default 0.1")
    ap.add_argument("--init-map", choices=["identity", "bottom", "spread"], default="identity",
                    help="--init-from into a bigger model (see init_partial): identity = copy matching layers, extra layers start as exact "
                         "identities and wider FFNs are zero-padded (function-preserving); bottom = extra layers random; spread = duplicate old layers")
    ap.add_argument("--freeze-init-steps", type=int, default=0,
                    help="with --init-from: for the first N steps train only params that were NOT copied (warm up new layers)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="auto = cuda if available else cpu")
    ap.add_argument("--batch-size", type=int, default=0, help="optional cap on sentences per batch (0 = only --max-tokens limits). "
                    "On GPU prefer raising --max-tokens; scale --lr ~sqrt(k) when batches get k x larger")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for training batches (0 = main process, the CPU default)")
    ap.add_argument("--save-every", type=int, default=0, help="also write last.pt every N steps (0 = only at evals); use on Colab")
    ap.add_argument("--dev-sets", default=None, help='comma list PATH:WEIGHT (default: <data>/dev.jsonl:1)')
    args = ap.parse_args()
    global BATCH_CAP; BATCH_CAP = args.batch_size
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = random.Random(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{args.data}/tokenizer.json"); CFG["vocab"] = tok.get_vocab_size()
    train = load(f"{args.data}/train.jsonl")
    dev_sets = []
    for spec in (args.dev_sets or f"{args.data}/dev.jsonl:1").split(","):
        path, wt = spec.rsplit(":", 1)
        dev_sets.append((os.path.basename(path).replace(".jsonl", ""), load(path), float(wt)))
    dev = dev_sets[0][1]
    wsum = sum(w for _, _, w in dev_sets)

    def evaluate_all():
        """Evaluate every dev set; return weighted (loss, CER, EM) and a printable per-set string."""
        res = [(n, evaluate(model, d, tok, args, amp), w) for n, d, w in dev_sets]
        agg = [sum(r[k] * w for _, r, w in res) / wsum for k in range(3)]
        txt = " | ".join(f"{n}: loss {r[0]:.4f} CER {r[1] * 100:.2f}% EM {r[2] * 100:.1f}%" for n, r, _ in res)
        return agg, txt

    if args.device == "auto": args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("--device cuda but no CUDA GPU is visible")
    dev_type = args.device; device = torch.device(dev_type)
    amp_dtype = None if args.no_bf16 else torch.bfloat16
    if dev_type == "cuda" and amp_dtype is not None and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16  # T4/V100 etc: no bf16 -> fp16 autocast + GradScaler
    if dev_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        print(f"[step2] device cuda: {torch.cuda.get_device_name(0)} amp={amp_dtype}", flush=True)
    amp = (lambda: torch.autocast(dev_type, dtype=amp_dtype)) if amp_dtype is not None else (lambda: torch.autocast(dev_type, enabled=False))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    last_path = f"{args.ckpt_dir}/last.pt"
    if args.resume and os.path.exists(last_path):
        CFG.update(torch.load(last_path, weights_only=False, map_location="cpu")["cfg"])
    else:
        if args.init_from:  # start from the init checkpoint's size, then apply CLI overrides
            CFG.update({k: v for k, v in torch.load(args.init_from, weights_only=False, map_location="cpu")["cfg"].items() if k != "vocab"})
        for k in ("d_model", "heads", "d_ff", "enc_layers", "dec_layers", "dropout"):
            if getattr(args, k) is not None: CFG[k] = getattr(args, k)
    assert CFG["d_model"] % CFG["heads"] == 0, "d_model must be divisible by heads"
    model = Seq2Seq(CFG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[step2] params: {n_params:,} ({n_params/1e6:.2f}M) cfg={CFG}", flush=True)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    step, epoch, best = 0, 0, (float("inf"), float("inf"))
    last = f"{args.ckpt_dir}/last.pt"
    steps_per_epoch = len(batches(train, args.max_tokens, True, random.Random(0)))
    total = int(args.epochs * steps_per_epoch)
    if args.sched == "cosine":
        def sched_fn(s):
            if s < args.warmup: return (s + 1) / args.warmup
            p = min(1.0, (s - args.warmup) / max(1, total - args.warmup))
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    else:
        sched_fn = lambda s: min((s + 1) / args.warmup, math.sqrt(args.warmup / (s + 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    def save(path, extra=None):
        ck = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, cfg=CFG, step=step, epoch=epoch, best=best)
        if extra: ck.update(extra)
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)

    if args.resume and os.path.exists(last):
        ck = torch.load(last, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])  # opt state follows param device
        if "scaler" in ck and scaler.is_enabled(): scaler.load_state_dict(ck["scaler"])
        step, epoch, best = ck["step"], ck["epoch"], tuple(ck["best"]); rng.setstate(ck["rng"])
        print(f"[step2] resumed from {last} at step {step}, epoch {epoch}, best={best}", flush=True)
    elif args.init_from:
        src_ck = torch.load(args.init_from, weights_only=False, map_location="cpu")
        model.cpu(); copied = init_partial(model, src_ck, args.init_map); model.to(device)
        print(f"[step2] fine-tuning from weights {args.init_from} (cfg {src_ck.get('cfg')}): copied {len(copied)}/{len(model.state_dict())} tensors", flush=True)
        if args.freeze_init_steps > 0 and len(copied) < len(model.state_dict()):
            frozen = [p for n, p in model.named_parameters() if n in copied]
            for p_ in frozen: p_.requires_grad_(False)
            print(f"[step2] freezing {len(frozen)} copied params for the first {args.freeze_init_steps} steps", flush=True)
        # baseline eval of the initial weights so 'best' is only replaced by a real improvement
        (dl, cer, em), txt = evaluate_all(); best = (cer, dl)
        print(f"[eval] step 0 (init) dev_loss {dl:.4f} dev_CER {cer * 100:.2f}% dev_EM {em * 100:.1f}% [{txt}]", flush=True)
        save(f"{args.ckpt_dir}/best.pt")

    print(f"[step2] train={len(train)} dev_sets={[(n, len(d), w) for n, d, w in dev_sets]} steps/epoch={steps_per_epoch} total_steps={total} device={dev_type} amp={amp_dtype} sched={args.sched} lr={args.lr}", flush=True)

    def resume_state():
        d = dict(opt=opt.state_dict(), sched=sched.state_dict(), rng=epoch_rng_state)
        if scaler.is_enabled(): d["scaler"] = scaler.state_dict()
        return d

    model.train(); t0, s0, run_loss, run_n = time.time(), step, 0.0, 0
    while step < total:
        epoch_rng_state = rng.getstate()
        order = batches(train, args.max_tokens, True, rng)
        skip = step - epoch * steps_per_epoch if step > epoch * steps_per_epoch else 0  # mid-epoch resume
        loader = torch.utils.data.DataLoader(_BatchDS(train, order[skip:]), batch_size=None, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=(dev_type == "cuda"),
                                             persistent_workers=False)
        for src, tgt, w in loader:
            if step >= total: break
            nb = dev_type == "cuda"
            src, tgt, w = src.to(device, non_blocking=nb), tgt.to(device, non_blocking=nb), w.to(device, non_blocking=nb)
            with amp():
                logits = model(src, tgt[:, :-1])
            tok_loss = F.cross_entropy(logits.float().transpose(1, 2), tgt[:, 1:], ignore_index=PAD,
                                       label_smoothing=0.1, reduction="none")          # [B, T]
            mask = (tgt[:, 1:] != PAD).float()
            loss = (tok_loss * w[:, None]).sum() / mask.sum()  # per-row weights; gold rows w=1
            if args.freeze_init_steps and step == args.freeze_init_steps:
                for p_ in model.parameters(): p_.requires_grad_(True)
                print(f"[step2] step {step}: unfroze all params", flush=True)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
            run_loss += loss.item(); run_n += 1
            if step % 50 == 0:
                sps = (step - s0) / (time.time() - t0); eta = (total - step) / max(sps, 1e-9)
                print(f"[train] step {step}/{total} ep {epoch + 1} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{sps:.2f} steps/s ETA {eta / 3600:.2f}h", flush=True); run_loss = run_n = 0
            if step % args.eval_every == 0 or step == total:
                te = time.time(); (dl, cer, em), txt = evaluate_all()
                improved = (cer, dl) < best
                if improved: best = (cer, dl); save(f"{args.ckpt_dir}/best.pt")
                save(last, resume_state())
                print(f"[eval] step {step} dev_loss {dl:.4f} dev_CER {cer * 100:.2f}% dev_EM {em * 100:.1f}% "
                      f"best_CER {best[0] * 100:.2f}% {'*saved best*' if improved else ''} ({time.time() - te:.0f}s) [{txt}]", flush=True)
                backup(args)
                t0, s0 = time.time(), step  # exclude eval time from speed
            elif args.save_every and step % args.save_every == 0:
                save(last, resume_state()); print(f"[save] step {step} -> {last}", flush=True)
        else:
            epoch += 1
    print(f"[step2] done. best dev CER {best[0] * 100:.2f}% dev loss {best[1]:.4f} -> {args.ckpt_dir}/best.pt", flush=True)
    backup(args)

if __name__ == "__main__":
    main()
