"""k4v4 모드용 채널 평균 표 생성 (Hugging Face 경로, vLLM 불필요)

    python -m kvcodec.calibrate --model Qwen/Qwen3-0.6B --text results/corpus_gutenberg.txt \\
        --out results/kv_mean_qwen3-0.6b.pt

- 레이어별 K/V의 (head, 채널) 평균을 토큰·배치에 대해 계산
- 캐시의 K는 RoPE 적용 후 값이라 평균이 위치 범위에 따라 달라짐 → 데모에서 쓸 문맥 길이만큼 긴 창으로 보정
- 저장 형식: {"k": (L, H, D), "v": (L, H, D), "seq_len": int, "n_tokens": int}
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-windows", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok(open(args.text, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    n = min(args.n_windows, ids.numel() // args.seq_len)
    sums = None
    for i in range(n):
        window = ids[i * args.seq_len : (i + 1) * args.seq_len][None].cuda()
        cache = DynamicCache(config=model.config)
        model(input_ids=window, past_key_values=cache, use_cache=True)
        cur = [torch.stack([l.keys.float().mean(dim=(0, 2)) for l in cache.layers]),
               torch.stack([l.values.float().mean(dim=(0, 2)) for l in cache.layers])]
        sums = cur if sums is None else [a + b for a, b in zip(sums, cur)]
    out = {"k": (sums[0] / n).cpu(), "v": (sums[1] / n).cpu(), "seq_len": args.seq_len,
           "n_tokens": n * args.seq_len}
    torch.save(out, args.out)
    print(f"-> {args.out}  k {tuple(out['k'].shape)}  windows {n} x {args.seq_len}")


if __name__ == "__main__":
    main()
