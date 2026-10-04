#!/usr/bin/env python
"""Path 2 of 3: the decode loop written out, reducing each step inside the loop.

Self-contained on purpose - the loop is copied here rather than imported from
src/dxo, so this script keeps measuring today's design even if that file changes.

    # Stop any vLLM server first: it reserves VRAM at startup, so "idle" is not
    # enough. With the dxo-orchestration compose file that is:
    #     docker compose --profile gpu down
    python run_manual.py

The one structural difference from run_generate.py: `del logits` sits INSIDE the
loop, so at most one full-vocab row is resident at a time. Everything else -
model, prompt, greedy decoding, the per-step reduction - is identical.

Greedy only. The real backend also does temperature/top-k/top-p; this experiment
fixes decoding to greedy so all three paths emit the same tokens, which is what
makes tokens/sec comparable.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "manual"
LABEL = "manual loop (reduce in-loop)"
MODEL = "Qwen/Qwen3-4B-FP8"
PROMPT = "Count from 1 to 300, separated by commas. Do not stop early."
BUDGETS = (32, 128, 512)
TOP_LOGPROBS = 3
RESULTS = Path(__file__).parent / "results"


def reduce_step(logits_row: torch.Tensor, chosen: int, tok) -> dict:
    """Full vocab row -> a handful of scalars. Same reduction in all three scripts."""
    logprobs = torch.log_softmax(logits_row.float(), dim=-1)
    probs = logprobs.exp()
    values, ids = logprobs.topk(TOP_LOGPROBS)
    return {
        "token": tok.decode([chosen]),
        "logprob": float(logprobs[chosen]),
        "entropy": float(-(probs * logprobs).nansum()),
        "top": [
            (tok.decode([int(t)]), float(v))
            for v, t in zip(values.tolist(), ids.tolist(), strict=True)
        ],
    }


def decode(model, tok, input_ids: torch.Tensor, budget: int, eos_ids: frozenset[int]) -> tuple:
    """One forward per token. First pass is the whole prompt, then one token each."""
    cur, cache = input_ids, None
    token_ids: list[int] = []
    steps: list[dict] = []

    with torch.inference_mode():
        for _ in range(budget):
            # logits_to_keep=1 is what generate() passes (generation/utils.py sets it
            # when forward() accepts it). Without it the prefill computes a logit row
            # for all 31 prompt positions against generate()'s 1 - ~8.7 MiB this path
            # would pay and the other would not. Fairness, not an optimization.
            out = model(
                input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = out.past_key_values
            logits = out.logits[0, -1].float()  # only the last position predicts

            chosen = int(logits.argmax())  # greedy
            steps.append(reduce_step(logits, chosen, tok))
            del out, logits  # <-- the whole point: one row resident, not `budget` rows

            token_ids.append(chosen)
            if chosen in eos_ids:
                break
            cur = torch.tensor([[chosen]], device=model.device, dtype=input_ids.dtype)

    return token_ids, steps


def other_gpu_mib() -> float:
    """GPU memory held by OTHER processes.

    Worth printing before measuring: a Jupyter kernel or a stray server sharing
    the card both skews the peak and can OOM a long run outright. The number goes
    into the result file so a surprising measurement can be explained later.
    """
    import os
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    mine, total = os.getpid(), 0.0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_text, mib_text = (part.strip() for part in line.split(","))
        if int(pid_text) != mine:
            total += float(mib_text)
    return total


def save(runs: list[dict], meta: dict) -> Path:
    """Write after every budget, not once at the end.

    A long run can OOM when the card is shared, and losing the budgets that
    already succeeded to a failure on the last one is avoidable.
    """
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{PATH}.json"
    out_path.write_text(
        json.dumps(
            {"path": PATH, "label": LABEL, "model": MODEL, "runs": runs, "meta": meta}, indent=2
        )
    )
    return out_path


def main() -> None:
    foreign = other_gpu_mib()
    print(f"other processes hold {foreign:.0f} MiB on this GPU")
    if foreign > 500:
        print("  WARNING: the card is shared. Peaks will be skewed and a long run may OOM.")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="cuda").eval()

    eos = model.generation_config.eos_token_id
    eos_ids = frozenset(eos if isinstance(eos, (list, tuple)) else [eos])

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_prompt = int(enc.input_ids.shape[1])

    meta = {
        "prompt_tokens": n_prompt,
        "vocab": int(model.config.vocab_size),
        "dtype": str(model.dtype),
        "decoding": "greedy",
        "other_gpu_mib_at_start": foreign,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    runs: list[dict] = []
    for budget in BUDGETS:
        if not runs:  # warm the kernels so the first timed run excludes Triton JIT
            decode(model, tok, enc.input_ids, 4, eos_ids)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated() / 2**20  # weights, before decoding
        torch.cuda.reset_peak_memory_stats()

        try:
            started = time.perf_counter()
            token_ids, steps = decode(model, tok, enc.input_ids, budget, eos_ids)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
        except torch.OutOfMemoryError:
            # Record the failure instead of losing the whole run to it. On a shared
            # card this says more about the neighbour than about the decode path.
            runs.append(
                {
                    "budget": budget,
                    "tokens": 0,
                    "latency_s": 0.0,
                    "tok_per_s": None,
                    "peak_mib": None,
                    "baseline_mib": baseline,
                    "decode_mib": None,
                    "note": f"OOM; other processes held {other_gpu_mib():.0f} MiB",
                }
            )
            print(f"budget={budget:>4} OOM (card shared?) - recorded and continuing")
            torch.cuda.empty_cache()
            save(runs, meta)
            continue

        peak = torch.cuda.max_memory_allocated() / 2**20
        runs.append(
            {
                "budget": budget,
                "tokens": len(token_ids),
                "latency_s": latency,
                "tok_per_s": len(token_ids) / latency,
                "peak_mib": peak,
                "baseline_mib": baseline,
                "decode_mib": peak - baseline,
                "note": "1 full-vocab logit row alive at peak",
            }
        )
        print(
            f"budget={budget:>4} tokens={runs[-1]['tokens']:>4} "
            f"{latency:>6.2f}s {runs[-1]['tok_per_s']:>6.1f} tok/s "
            f"decode={runs[-1]['decode_mib']:>8.1f} MiB"
        )
        save(runs, meta)

    print(f"-> {save(runs, meta)}")


if __name__ == "__main__":
    main()
