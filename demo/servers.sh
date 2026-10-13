#!/usr/bin/env bash
# KV 캐시 코덱 비교용 vLLM 서버 5개 실행 (docker, GPU 1장 공유)
#   demo/servers.sh up [이름...] | down [이름...] | status
#   이름을 주면 그 서버만 (bf16 vllm-int8 vllm-int4 ours-k8v4 ours-k4v4). 5개 전부는 GPU 약 10~11GB 필요
# - 모든 서버가 같은 모델, 같은 KV 캐시 크기(POOL_MB)를 사용 → 코덱에 따라 수용 토큰 수만 달라짐
# - 조건 통일: --enforce-eager, prefix caching 끔
set -euo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai:v0.30.0}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
POOL_MB=${POOL_MB:-256}
MAX_LEN=${MAX_LEN:-2048}   # bf16이 POOL_MB=256에 담을 수 있는 한도(2,336토큰) 이내
GPU_UTIL=${GPU_UTIL:-0.1}   # 시작 시 여유 메모리 검사 기준. KV 캐시 크기는 POOL_MB로 따로 고정
REPO=$(cd "$(dirname "$0")/.." && pwd)
MEAN=/repo/results/kv_mean_qwen3-0.6b.pt

# 이름 | 포트 | 환경변수 | vllm 인자
SERVERS=(
  "bf16|8101||--attention-backend TRITON_ATTN"
  "vllm-int8|8102||--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  "vllm-int4|8103||--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head"
  "ours-k8v4|8104|-e KVCODEC=k8v4|--attention-backend CUSTOM"
  "ours-k4v4|8105|-e KVCODEC=k4v4 -e KVCODEC_MEAN=$MEAN|--attention-backend CUSTOM"
)

wanted() {  # 인자가 없으면 전부, 있으면 그 이름만
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

up() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name port envs args <<<"$spec"
    wanted "$name" || continue
    if docker ps --format '{{.Names}}' | grep -qx "kvdemo-$name"; then
      echo "kvdemo-$name: 이미 실행 중"; continue
    fi
    docker rm -f "kvdemo-$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "kvdemo-$name" --gpus all --ipc=host -p "127.0.0.1:$port:8000" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $envs \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$REPO:/repo:ro" \
      --entrypoint bash "$IMAGE" -c "cp -r /repo /tmp/r && pip install -q --no-deps --no-build-isolation /tmp/r && \
        exec vllm serve $MODEL --served-model-name $name --kv-cache-memory-bytes $((POOL_MB * 1024 * 1024)) \
        --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_LEN --max-num-seqs 4 --enforce-eager --no-enable-prefix-caching $args" >/dev/null
    printf "kvdemo-%s: 시작 대기" "$name"
    for _ in $(seq 1 120); do
      if curl -sf "localhost:$port/health" >/dev/null 2>&1; then echo " -> 준비 완료 (:$port)"; break; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "kvdemo-$name"; then
        echo " -> 실패"; docker logs --tail 15 "kvdemo-$name" 2>&1 | sed 's/^/    /'; break
      fi
      printf "."; sleep 2
    done
  done
}

down() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name _ <<<"$spec"
    wanted "$name" || continue
    docker rm -f "kvdemo-$name" >/dev/null 2>&1 && echo "kvdemo-$name: 중지" || true
  done
}

status() {
  for spec in "${SERVERS[@]}"; do
    IFS='|' read -r name port _ <<<"$spec"
    if curl -sf "localhost:$port/health" >/dev/null 2>&1; then echo "kvdemo-$name :$port  실행 중"; else echo "kvdemo-$name :$port  꺼짐"; fi
  done
}

ONLY=("${@:2}")
case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "usage: $0 up [name...] | down [name...] | status"; exit 1 ;;
esac
