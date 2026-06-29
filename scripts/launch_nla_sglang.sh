#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-kitft/nla-qwen2.5-7b-L20-av}"
PORT="${2:-30000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if ! python -c 'import sglang' >/dev/null 2>&1; then
  echo 'Missing sglang. Install the platform-appropriate server package, for example:' >&2
  echo '  pip install "sglang[all]>=0.5.6"' >&2
  exit 2
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --disable-radix-cache
