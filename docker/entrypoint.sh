#!/bin/sh
# Pull the teencode model from Hugging Face (once, cached in $MODEL_ROOT), then run step_4_inference.
# No args          -> HTTP server on 0.0.0.0:8080
# Args after "--"  -> passed to step_4_inference (CLI mode: sentences as args, or stdin with -i)
set -eu

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_REVISION="${HF_REVISION:-main}"
MODEL_SUBDIR="${MODEL_SUBDIR:-onnx}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
THREADS="${THREADS:-4}"
FILES="${MODEL_FILES:-encoder.int8.onnx decoder.int8.onnx tokenizer.json output_vocab.txt config.json guard.json}"

die() { echo "[entrypoint] ERROR: $*" >&2; exit 1; }
log() { echo "[entrypoint] $*" >&2; }

[ -n "${HF_REPO:-}" ] || die "HF_REPO is not set. Use e.g. -e HF_REPO=<your-hf-username>/teencode-normalizer-v3"
case "$HF_REPO" in */*) ;; *) die "HF_REPO must look like <user>/<repo>, got '$HF_REPO'";; esac

SAFE=$(echo "$HF_REPO@$HF_REVISION" | tr '/:@' '___')
DIR="$MODEL_ROOT/$SAFE/$MODEL_SUBDIR"
mkdir -p "$DIR" || die "cannot create $DIR (is the /models volume writable?)"

fetch() {
  f="$1"; url="$HF_ENDPOINT/$HF_REPO/resolve/$HF_REVISION/$MODEL_SUBDIR/$f"
  if [ -s "$DIR/$f" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
    log "cached  $f"; return 0
  fi
  log "download $url"
  tmp="$DIR/.$f.part"
  if [ -n "${HF_TOKEN:-}" ]; then
    code=$(curl -sSL --retry 3 -H "Authorization: Bearer $HF_TOKEN" -o "$tmp" -w '%{http_code}' "$url") || code=000
  else
    code=$(curl -sSL --retry 3 -o "$tmp" -w '%{http_code}' "$url") || code=000
  fi
  case "$code" in
    200) mv -f "$tmp" "$DIR/$f" ;;
    401|403) rm -f "$tmp"; die "HTTP $code for $url - repo is private/gated or HF_TOKEN is wrong. Set HF_TOKEN=hf_... with read access." ;;
    404) rm -f "$tmp"; die "HTTP 404 for $url - check HF_REPO='$HF_REPO', HF_REVISION='$HF_REVISION', MODEL_SUBDIR='$MODEL_SUBDIR' (file must be at $MODEL_SUBDIR/$f in the repo)." ;;
    000) rm -f "$tmp"; die "network error fetching $url (no internet / DNS / proxy?)" ;;
    *) rm -f "$tmp"; die "HTTP $code for $url" ;;
  esac
}

for f in $FILES; do fetch "$f"; done
log "model ready in $DIR"

if [ "${1:-}" = "--" ]; then
  shift   # CLI mode: "--" alone = read sentences from stdin (docker run -i)
elif [ "$#" -eq 0 ]; then
  set -- --serve "${LISTEN:-0.0.0.0:8080}"
fi
exec step_4_inference --model-dir "$DIR" --threads "$THREADS" "$@"
