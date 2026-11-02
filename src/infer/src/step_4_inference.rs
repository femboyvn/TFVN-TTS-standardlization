//! STEP 4 — Inference with ONNX Runtime (pure Rust, runs on Raspberry Pi 4 / aarch64)
//! ---------------------------------------------------------------------------------
//! Reads teencode sentences from the command line (each arg = one sentence) or, if none, from stdin
//! (one per line) and prints the normalized standard Vietnamese.
//!
//! Pipeline per sentence:
//!   1. BPE-encode, run the encoder once.
//!   2. Decode: greedy, or beam search (default beam 4, GNMT length penalty ((5+len)/6)^alpha).
//!      No KV cache: the decoder re-reads the prefix each step (cheap for ~20-token sentences).
//!   3. Over-correction guard (settings in <model-dir>/guard.json, tuned on DEV by step_4_eval.py):
//!      - LCS word-alignment guard: keep the INPUT words except small local rewrites
//!        (<= align_max_in input words -> 1..align_max_out output words); pure insertions/deletions and long
//!        rewrites (hallucinated / repeated / dropped words) are rejected. Same algorithm as step_4_eval.py.
//!      - retone (untoned input): alignment keys are diacritic-stripped, so re-toning is a match, not an edit.
//!      - align_one_to_one: mismatched spans with equal word counts (word-by-word substitutions) are accepted.
//!      - optional word repair: output word not in training-output vocab (output_vocab.txt) -> input word.
//!      - optional sentence fallback: edit-distance ratio > max_edit or bad length ratio -> return input.
//!
//! `ort` is built with `load-dynamic`: set ORT_DYLIB_PATH to libonnxruntime.so (pip wheel's
//! onnxruntime/capi/libonnxruntime.so.*, or the aarch64 release for the Pi).
//!
//! Usage: ORT_DYLIB_PATH=... step_4_inference [--model-dir onnx] [--fp32] [--greedy] [--beam N] [--no-guard] [--threads 4] [--time] ["sentence" ...]
//! Model size (layers, d_model, d_ff) is read from the ONNX graphs, so any step-3 export works unchanged.
use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::{session::Session, value::Tensor};
use std::collections::HashSet;
use std::io::BufRead;
use tokenizers::Tokenizer;

const BOS: i64 = 1;
const EOS: i64 = 2;
const DEFAULT_MAX_LEN: usize = 192; // used when <model-dir>/config.json (written by step 3) is missing

/// Guard settings (onnx/guard.json written by step_4_eval.py). Defaults = no guard.
#[derive(serde::Deserialize, Default, Clone)]
#[serde(default)]
struct Guard {
    beam: usize,
    alpha: f32,
    word_repair: bool,
    max_edit: Option<f32>,
    align_max_in: usize,
    align_max_out: usize,
    align_one_to_one: bool,
    retone: bool,
    min_len_ratio: f32,
    max_len_ratio: f32,
}

struct Normalizer { enc: Session, dec: Session, tok: Tokenizer, g: Guard, vocab: HashSet<String>, max_len: usize }

/// Word-level LCS alignment -> segments (i1, i2, j1, j2, equal). Mirrors lcs_segments() in step_4_eval.py.
fn lcs_segments(a: &[&str], b: &[&str]) -> Vec<(usize, usize, usize, usize, bool)> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() { for j in (0..m).rev() {
        dp[i][j] = if a[i] == b[j] { dp[i + 1][j + 1] + 1 } else { dp[i + 1][j].max(dp[i][j + 1]) };
    } }
    let (mut segs, mut i, mut j, mut si, mut sj) = (vec![], 0, 0, 0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            if (si, sj) != (i, j) { segs.push((si, i, sj, j, false)); }
            segs.push((i, i + 1, j, j + 1, true)); i += 1; j += 1; si = i; sj = j;
        } else if dp[i + 1][j] >= dp[i][j + 1] { i += 1 } else { j += 1 }
    }
    if (si, sj) != (n, m) { segs.push((si, n, sj, m, false)); }
    segs
}

/// Remove Vietnamese diacritics (NFD + drop combining marks, đ->d). Mirrors strip_diacritics() in step_4_eval.py.
fn strip_diacritics(w: &str) -> String {
    use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
    w.nfd().filter(|c| !is_combining_mark(*c)).map(|c| match c { 'đ' => 'd', 'Đ' => 'D', x => x }).collect()
}

fn char_edit(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i];
        for j in 1..=b.len() { cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + (a[i - 1] != b[j - 1]) as usize)); }
        prev = cur;
    }
    prev[b.len()]
}

impl Normalizer {
    fn load(dir: &str, fp32: bool, threads: usize) -> Result<Self> {
        let sfx = if fp32 { "onnx" } else { "int8.onnx" };
        let mk = |name: &str| -> Result<Session> {
            Ok(Session::builder()?.with_intra_threads(threads)?.commit_from_file(format!("{dir}/{name}.{sfx}"))?)
        };
        let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).map_err(|e| anyhow!(e))?;
        let g: Guard = std::fs::read_to_string(format!("{dir}/guard.json")).ok()
            .and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let vocab = std::fs::read_to_string(format!("{dir}/output_vocab.txt")).unwrap_or_default()
            .lines().map(String::from).collect();
        // model size is baked into the ONNX graphs; only max_len is needed here (config.json from step 3)
        let max_len = std::fs::read_to_string(format!("{dir}/config.json")).ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v["max_len"].as_u64()).map(|x| x as usize).unwrap_or(DEFAULT_MAX_LEN);
        Ok(Self { enc: mk("encoder")?, dec: mk("decoder")?, tok, g, vocab, max_len })
    }

    /// log-softmax over the last-position logits for prefix `ys`
    fn step(&mut self, ys: &[i64], mem: &Array3<f32>, src: &Array2<i64>) -> Result<Vec<f32>> {
        let tgt = Array2::from_shape_vec((1, ys.len()), ys.to_vec())?;
        let out = self.dec.run(ort::inputs![
            "tgt_ids" => Tensor::from_array(tgt)?,
            "memory" => Tensor::from_array(mem.clone())?,
            "src_ids" => Tensor::from_array(src.clone())?])?;
        let (_, l) = out["logits"].try_extract_tensor::<f32>()?;
        let mx = l.iter().cloned().fold(f32::MIN, f32::max);
        let lse = mx + l.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
        Ok(l.iter().map(|v| v - lse).collect())
    }

    fn decode(&mut self, text: &str, beam: usize) -> Result<String> {
        let e = self.tok.encode(text, false).map_err(|e| anyhow!(e))?;
        let mut src: Vec<i64> = vec![BOS];
        src.extend(e.get_ids().iter().take(self.max_len - 2).map(|&x| x as i64));
        src.push(EOS);
        let s = src.len();
        let src = Array2::from_shape_vec((1, s), src)?;
        let out = self.enc.run(ort::inputs!["src_ids" => Tensor::from_array(src.clone())?])?;
        let (ms, md) = out["memory"].try_extract_tensor::<f32>()?;
        let mem = Array3::from_shape_vec((ms[0] as usize, ms[1] as usize, ms[2] as usize), md.to_vec())?;
        drop(out);
        let max_steps = (s + 24).min(self.max_len - 1);
        let lp = |len: usize, a: f32| ((5.0 + len as f32) / 6.0).powf(a);
        let mut beams: Vec<(Vec<i64>, f32)> = vec![(vec![BOS], 0.0)];
        let mut done: Vec<(Vec<i64>, f32)> = vec![];
        for _ in 0..max_steps {
            let mut cand = vec![];
            for (seq, sc) in &beams {
                let l = self.step(seq, &mem, &src)?;
                let mut idx: Vec<usize> = (0..l.len()).collect();
                idx.select_nth_unstable_by(beam - 1, |&x, &y| l[y].partial_cmp(&l[x]).unwrap());
                for &t in &idx[..beam] { let mut q = seq.clone(); q.push(t as i64); cand.push((q, sc + l[t])); }
            }
            cand.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
            beams.clear();
            for (q, sc) in cand {
                if *q.last().unwrap() == EOS { let n = q.len(); done.push((q, sc / lp(n, self.g.alpha))); }
                else { beams.push((q, sc)); }
                if beams.len() == beam { break; }
            }
            if done.len() >= beam || beams.is_empty() { break; }
        }
        if done.is_empty() { done = beams.into_iter().map(|(q, sc)| { let n = q.len(); (q, sc / lp(n, self.g.alpha)) }).collect(); }
        let best = done.into_iter().max_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap().0;
        let ids: Vec<u32> = best.iter().filter(|&&t| t != BOS && t != EOS).map(|&x| x as u32).collect();
        self.tok.decode(&ids, true).map_err(|e| anyhow!(e))
    }

    /// Post-processing guard against over-correction (see module docs).
    fn guard(&self, inp: &str, out: String) -> String {
        let g = &self.g;
        let mut out = out;
        if g.align_max_in > 0 {
            let (a, b): (Vec<&str>, Vec<&str>) = (inp.split_whitespace().collect(), out.split_whitespace().collect());
            let mut res: Vec<&str> = vec![];
            // retone: align on diacritic-stripped keys so re-toned untoned words ("khong" -> "không") count as matches
            let (ka, kb): (Vec<String>, Vec<String>) = if g.retone {
                (a.iter().map(|w| strip_diacritics(w)).collect(), b.iter().map(|w| strip_diacritics(w)).collect())
            } else { (a.iter().map(|w| w.to_string()).collect(), b.iter().map(|w| w.to_string()).collect()) };
            let (ka, kb): (Vec<&str>, Vec<&str>) = (ka.iter().map(|x| x.as_str()).collect(), kb.iter().map(|x| x.as_str()).collect());
            for (i1, i2, j1, j2, eq) in lcs_segments(&ka, &kb) {
                let small = i2 > i1 && i2 - i1 <= g.align_max_in && j2 > j1 && j2 - j1 <= g.align_max_out;
                let one_to_one = g.align_one_to_one && i2 - i1 == j2 - j1;
                if eq || small || one_to_one { res.extend(&b[j1..j2]) } else { res.extend(&a[i1..i2]) }
            }
            out = res.join(" ");
        }
        if g.word_repair {
            let (a, b): (Vec<&str>, Vec<&str>) = (inp.split_whitespace().collect(), out.split_whitespace().collect());
            if a.len() == b.len() {
                out = a.iter().zip(&b).map(|(i, o)| if self.vocab.contains(*o) || i == o
                    || (g.retone && strip_diacritics(i) == strip_diacritics(o)) { *o } else { *i })
                    .collect::<Vec<_>>().join(" ");
            }
        }
        if let Some(me) = g.max_edit {
            let n = inp.chars().count().max(1) as f32;
            let r = out.chars().count() as f32 / n;
            if char_edit(inp, &out) as f32 / n > me || r < g.min_len_ratio || r > g.max_len_ratio { return inp.to_string(); }
        }
        out
    }
}

fn main() -> Result<()> {
    let (mut dir, mut fp32, mut use_guard, mut beam_cli) = ("onnx".to_string(), false, true, None::<usize>);
    let (mut threads, mut timing, mut serve) = (4usize, false, None::<String>);
    let mut texts = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model-dir" => dir = args.next().expect("--model-dir needs a value"),
            "--fp32" => fp32 = true,
            "--greedy" => beam_cli = Some(1),
            "--beam" => beam_cli = Some(args.next().expect("--beam N").parse()?),
            "--no-guard" => use_guard = false,
            "--threads" => threads = args.next().expect("--threads N").parse()?,
            "--time" => timing = true,
            "--serve" => serve = Some(args.next().expect("--serve ADDR e.g. 0.0.0.0:8080")),
            _ => texts.push(a),
        }
    }
    let mut nz = Normalizer::load(&dir, fp32, threads)?;
    let beam = beam_cli.unwrap_or(nz.g.beam.max(1));
    let mut run = |t: &str| -> Result<String> {
        let t0 = std::time::Instant::now();
        let o = nz.decode(t, beam)?;
        if timing { eprintln!("[time] {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
        Ok(if use_guard { nz.guard(t, o) } else { o })
    };
    if let Some(addr) = serve {
        return serve_http(&addr, &mut run);
    }
    if texts.is_empty() {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if !line.trim().is_empty() { println!("{}", run(line.trim())?); }
        }
    } else {
        for t in texts { println!("{}", run(&t)?); }
    }
    Ok(())
}

/// Minimal HTTP mode: POST /normalize {"text": "..."} -> {"input": ..., "output": ...};
/// also accepts {"texts": [...]} -> {"outputs": [...]}. GET /health -> ok. Requests are served sequentially.
fn serve_http(addr: &str, run: &mut dyn FnMut(&str) -> Result<String>) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow!("bind {addr}: {e}"))?;
    eprintln!("[serve] listening on http://{addr}  (POST /normalize, GET /health)");
    let json_hdr = tiny_http::Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap();
    for mut req in server.incoming_requests() {
        let (code, body) = match (req.method().as_str(), req.url()) {
            ("GET", "/health") => (200, serde_json::json!({"status": "ok"})),
            ("POST", "/normalize") => {
                let mut buf = String::new();
                let _ = std::io::Read::read_to_string(req.as_reader(), &mut buf);
                match serde_json::from_str::<serde_json::Value>(&buf) {
                    Ok(v) => {
                        if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                            let t = t.trim();
                            match if t.is_empty() { Ok(String::new()) } else { run(t) } {
                                Ok(o) => (200, serde_json::json!({"input": t, "output": o})),
                                Err(e) => (500, serde_json::json!({"error": e.to_string()})),
                            }
                        } else if let Some(arr) = v.get("texts").and_then(|x| x.as_array()) {
                            let mut outs = vec![];
                            let mut err = None;
                            for t in arr.iter().filter_map(|x| x.as_str()) {
                                let t = t.trim();
                                if t.is_empty() { outs.push(String::new()); continue; }
                                match run(t) { Ok(o) => outs.push(o), Err(e) => { err = Some(e); break; } }
                            }
                            match err { None => (200, serde_json::json!({"outputs": outs})),
                                Some(e) => (500, serde_json::json!({"error": e.to_string()})) }
                        } else { (400, serde_json::json!({"error": "expected {\"text\": ...} or {\"texts\": [...]}"})) }
                    }
                    Err(e) => (400, serde_json::json!({"error": format!("bad json: {e}")})),
                }
            }
            _ => (404, serde_json::json!({"error": "use POST /normalize or GET /health"})),
        };
        let resp = tiny_http::Response::from_string(body.to_string()).with_status_code(code).with_header(json_hdr.clone());
        let _ = req.respond(resp);
    }
    Ok(())
}
