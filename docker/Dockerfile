# syntax=docker/dockerfile:1
# Teencode v3 normalizer, model PULLED from Hugging Face at container start (cached in the /models volume).
# Built by .github/workflows/docker.yml (amd64+arm64 -> ghcr.io). Rust runner from src/infer. Build context = REPO ROOT.
#   docker build -f docker/Dockerfile -t teencode-hf .
#   docker buildx build --platform linux/arm64 -f docker/Dockerfile -t teencode-hf:arm64 --load .
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t <user>/teencode-hf --push .
ARG ORT_VERSION=1.22.0

FROM rust:1-bookworm AS build
WORKDIR /src
COPY src/infer/Cargo.toml src/infer/Cargo.lock ./
COPY src/infer/src ./src
RUN cargo build --release --locked --bin step_4_inference \
 && cp target/release/step_4_inference /step_4_inference

FROM debian:bookworm-slim AS ort
ARG ORT_VERSION
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      amd64) ORT_ARCH=x64 ;; arm64) ORT_ARCH=aarch64 ;; \
      *) echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz"; \
    tar xzf /tmp/ort.tgz -C /tmp; mkdir -p /ort; \
    cp -L "/tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" /ort/libonnxruntime.so

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -r -u 10001 -m teencode && mkdir -p /models && chown teencode /models
COPY --from=ort   /ort/libonnxruntime.so /opt/onnxruntime/lib/libonnxruntime.so
COPY --from=build /step_4_inference      /usr/local/bin/step_4_inference
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip CRs in case Windows git converted the script to CRLF.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod 755 /usr/local/bin/entrypoint.sh
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so LANG=C.UTF-8 \
    HF_ENDPOINT=https://huggingface.co HF_REVISION=main MODEL_SUBDIR=onnx MODEL_ROOT=/models THREADS=4
VOLUME /models
USER teencode
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
