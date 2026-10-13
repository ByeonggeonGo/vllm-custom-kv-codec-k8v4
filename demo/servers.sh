#!/usr/bin/env bash
# Start the 5 vLLM servers for the KV cache codec comparison (docker, one shared GPU)
#   demo/servers.sh up [name...] | down [name...] | status
#   with names, only those servers (bf16 vllm-int8 vllm-int4 ours-k8v4 ours-k4v4). All 5 need about 12 GB of GPU memory
# - every server uses the same model and the same KV cache size (POOL_MB) -> only the token capacity differs by codec
# - same conditions: --enforce-eager, prefix caching off
set -euo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:v0.30.0}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
POOL_MB=${POOL_MB:-256}
MAX_LEN=${MAX_LEN:-2048}   # within what bf16 can hold in POOL_MB=256 (2,336 tokens)
GPU_UTIL=${GPU_UTIL:-0.1}   # threshold of the free-memory check at startup. The KV cache size is fixed separately by POOL_MB
REPO=$(cd "$(dirname "$0")/.." && pwd)
MEAN=/repo/results/kv_mean_qwen3-0.6b.pt

# name | port | environment | vllm arguments
SERVERS=(
  "bf16|8101||--attention-backend TRITON_ATTN"
  "vllm-int8|8102||--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  "vllm-int4|8103||--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head"
  "ours-k8v4|8104|-e KVCODEC=k8v4|--attention-backend CUSTOM"
  "ours-k4v4|8105|-e KVCODEC=k4v4 -e KVCODEC_MEAN=$MEAN|--attention-backend CUSTOM"
)

wanted() {  # no arguments: all servers. Otherwise only the named ones
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

up() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name port envs args <<<"$spec"
    wanted "$name" || continue
    if docker ps --format '{{.Names}}' | grep -qx "kvdemo-$name"; then
      echo "kvdemo-$name: already running"; continue
    fi
    docker rm -f "kvdemo-$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "kvdemo-$name" --gpus all --ipc=host -p "127.0.0.1:$port:8000" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $envs \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$REPO:/repo:ro" \
      --entrypoint bash "$IMAGE" -c "cp -r /repo /tmp/r && pip install -q --no-deps --no-build-isolation /tmp/r && \
        exec vllm serve $MODEL --served-model-name $name --kv-cache-memory-bytes $((POOL_MB * 1024 * 1024)) \
        --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_LEN --max-num-seqs 4 --enforce-eager --no-enable-prefix-caching $args" >/dev/null
    printf "kvdemo-%s: starting" "$name"
    for _ in $(seq 1 120); do
      if curl -sf "localhost:$port/health" >/dev/null 2>&1; then echo " -> ready (:$port)"; break; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "kvdemo-$name"; then
        echo " -> failed"; docker logs --tail 15 "kvdemo-$name" 2>&1 | sed 's/^/    /'; break
      fi
      printf "."; sleep 2
    done
  done
}

down() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name _ <<<"$spec"
    wanted "$name" || continue
    docker rm -f "kvdemo-$name" >/dev/null 2>&1 && echo "kvdemo-$name: stopped" || true
  done
}

status() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name port _ <<<"$spec"
    if curl -sf "localhost:$port/health" >/dev/null 2>&1; then echo "kvdemo-$name :$port  up"; else echo "kvdemo-$name :$port  down"; fi
  done
}

ONLY=("${@:2}")
case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up [name...] | down [name...] | status"; exit 1 ;;
esac
