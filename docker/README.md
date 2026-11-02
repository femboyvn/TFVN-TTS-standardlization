# Docker: model pulled from Hugging Face

The image contains only the Rust runner (`step_4_inference`, built from `src/infer`) and ONNX Runtime 1.22.0. **The model is not baked in.** When the container starts, `entrypoint.sh` downloads the INT8 model from `HF_REPO` into the `/models` volume. It does this once and reuses the cached copy after that. Then it starts the HTTP server (the default) or CLI mode.

**The image is built by GitHub Actions** (`.github/workflows/docker.yml`) for linux/amd64 and linux/arm64 (Raspberry Pi 4, 64-bit OS). It is pushed to `ghcr.io/<github-user>/teencode-normalizer`:
- `:latest` comes from `main`.
- `:X.Y.Z` / `:X.Y` come from `vX.Y.Z` tags.
- `:sha-…` is pushed for every build.

The first workflow run creates a **private** package. To allow pulls without logging in, open GitHub → your profile → Packages → teencode-normalizer → Package settings → Change visibility → Public.

Files expected in the HF model repo: `onnx/encoder.int8.onnx`, `decoder.int8.onnx`, `tokenizer.json`, `output_vocab.txt`, `config.json`, `guard.json`.

## Run (Windows PowerShell, Linux, Raspberry Pi)
```powershell
docker pull ghcr.io/<github-user>/teencode-normalizer:latest
docker run -d --name teencode -p 8080:8080 -v teencode-models:/models `
  -e HF_REPO=<hf-user>/teencode-normalizer-v3 --restart unless-stopped ghcr.io/<github-user>/teencode-normalizer:latest
docker logs -f teencode            # first start: "download ...", later "cached ..."
```
On Linux or the Pi, replace the PowerShell backtick with `\`.

CLI mode (everything after `--` goes to the runner):
```powershell
docker run --rm -v teencode-models:/models -e HF_REPO=<hf-user>/teencode-normalizer-v3 ghcr.io/<github-user>/teencode-normalizer -- "c oi, e nho c qua"
"thang lam ML" | docker run --rm -i -v teencode-models:/models -e HF_REPO=<hf-user>/teencode-normalizer-v3 ghcr.io/<github-user>/teencode-normalizer --
```
For piped Vietnamese text in Windows PowerShell 5.1, run `$OutputEncoding = [Text.UTF8Encoding]::new()` first.

HTTP:
```powershell
$body = [Text.Encoding]::UTF8.GetBytes((@{ text = "c oi, e nho c qua" } | ConvertTo-Json))
$r = Invoke-WebRequest http://localhost:8080/normalize -Method Post -ContentType "application/json; charset=utf-8" -Body $body
[Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json     # -> chị ơi, em nhớ chị quá
curl.exe http://localhost:8080/health
```

## Settings (env)
| Var | Default | Meaning |
|---|---|---|
| `HF_REPO` | *(required)* | `<hf-user>/teencode-normalizer-v3` |
| `HF_REVISION` | `main` | branch / tag / commit |
| `HF_TOKEN` | – | read token, only needed for private repos |
| `HF_ENDPOINT` | `https://huggingface.co` | mirror / proxy |
| `MODEL_SUBDIR` | `onnx` | folder in the repo |
| `THREADS` | `4` | ONNX Runtime threads |
| `LISTEN` | `0.0.0.0:8080` | HTTP bind address |
| `FORCE_DOWNLOAD` | `0` | `1` = download again even if cached |

## Local build (secondary: CI does this for you)
```powershell
docker build -f docker/Dockerfile -t teencode .                          # from the repo root
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t teencode:arm64 --load .
cd docker; Copy-Item .env.example .env; docker compose up -d --build      # set HF_REPO in .env
```

## Errors
- `HF_REPO is not set`: pass `-e HF_REPO=...`.
- `HTTP 404`: the repo or revision is wrong, or the files are not under `onnx/`.
- `HTTP 401/403`: the repo is private, so set `HF_TOKEN`.
- To pick up a new model after you upload it, run once with `-e FORCE_DOWNLOAD=1`, or remove the volume with `docker volume rm teencode-models`.
