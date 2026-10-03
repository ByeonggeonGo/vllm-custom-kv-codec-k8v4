#!/usr/bin/env python
"""Path 1 of 3: transformers' own `model.generate(output_logits=True)`.

Self-contained on purpose - imports nothing from this repo, so it keeps measuring
what it measured today even if src/dxo changes later.

    # Stop any vLLM server first: it reserves VRAM at startup, so "idle" is not
    # enough. With the dxo-orchestration compose file that is:
    #     docker compose --profile gpu down
    python run_generate.py

What this path does: generate() accumulates one (1, vocab) logit row per step and
returns all of them, so the per-step reduction can only happen after the loop has
finished. That is the thing under measurement.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "generate"
LABEL = "generate() + output_logits"
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
        # Warm the kernels once so the first timed run is not paying for Triton JIT.
        if not runs:
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=4, do_sample=False)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated() / 2**20  # weights, before decoding
        torch.cuda.reset_peak_memory_stats()

        try:
            started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=budget,
                    do_sample=False,  # greedy: all three paths then emit identical tokens
                    return_dict_in_generate=True,
                    output_logits=True,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - started

            new_ids = out.sequences[0][n_prompt:]
            # The reduction happens HERE, after the loop - every row is still resident.
            steps = [reduce_step(row[0], int(new_ids[i]), tok) for i, row in enumerate(out.logits)]
            peak = torch.cuda.max_memory_allocated() / 2**20
            rows_held = len(out.logits)
            del out
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

        runs.append(
            {
                "budget": budget,
                "tokens": len(steps),
                "latency_s": latency,
                "tok_per_s": len(steps) / latency,
                "peak_mib": peak,
                "baseline_mib": baseline,
                "decode_mib": peak - baseline,
                "note": f"{rows_held} full-vocab logit rows alive at peak",
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
