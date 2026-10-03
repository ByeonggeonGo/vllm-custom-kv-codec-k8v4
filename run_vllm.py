#!/usr/bin/env python
"""Path 3 of 3: the vLLM server over HTTP.

Self-contained on purpose - talks to the server with plain HTTP, so it imports
nothing from this repo and not even the openai SDK.

    # Needs a vLLM server on localhost:8000 serving as "local". With the
    # dxo-orchestration compose file that is:
    #     docker compose --profile gpu up -d
    python run_vllm.py

Read the memory column with care. The other two paths hold the weights in their
own process, so `peak - baseline` isolates what decoding cost. Here the allocator
is in another process and vLLM claims --gpu-memory-utilization of the card at
STARTUP, mostly as KV cache, then works inside that slab. So:

  * `baseline_mib` / `peak_mib` are the server process's GPU footprint sampled
    before and during the request, via nvidia-smi.
  * `decode_mib` is therefore expected to be ~0, and that is not vLLM winning -
    it means the cost was paid up front and is invisible per request. The number
    is recorded to make the structural difference visible, not to rank the paths.

The honest comparison this script contributes is SPEED.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PATH = "vllm"
LABEL = "vLLM server (HTTP)"
MODEL = "local"  # --served-model-name
BASE_URL = "http://localhost:8000/v1"
PROMPT = "Count from 1 to 300, separated by commas. Do not stop early."
BUDGETS = (32, 128, 512)
TOP_LOGPROBS = 3
RESULTS = Path(__file__).parent / "results"


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def server_gpu_mib() -> tuple[int | None, float | None]:
    """The vLLM process's GPU footprint. torch cannot see another process's allocator."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    best = (None, None)
    for line in out.strip().splitlines():
        pid_text, mib_text = (part.strip() for part in line.split(","))
        pid = int(pid_text)
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue  # a container process this user cannot inspect
        # Case-insensitive: vLLM renames its process title to "VLLM::EngineCore",
        # so a lowercase substring test silently finds nothing.
        if "vllm" in cmdline.lower():
            best = (pid, float(mib_text))
    return best


class Sampler(threading.Thread):
    """Poll the server's GPU footprint while a request is in flight."""

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        # NOT self._stop: threading.Thread already defines a private _stop()
        # method that join() calls, and shadowing it breaks join().
        self.interval, self.samples, self._done = interval, [], threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            _, mib = server_gpu_mib()
            if mib is not None:
                self.samples.append(mib)
            self._done.wait(self.interval)

    def stop(self) -> float | None:
        self._done.set()
        self.join(timeout=2)
        return max(self.samples) if self.samples else None


def main() -> None:
    models = json.loads(urllib.request.urlopen(BASE_URL + "/models", timeout=10).read())
    served = [m["id"] for m in models["data"]]
    print(f"server up, serving {served}")

    pid, baseline = server_gpu_mib()
    print(f"vLLM pid={pid} holds {baseline} MiB before any request (preallocated)")

    # Warm the server so the first timed request is not paying for a cold cache.
    post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    runs = []
    for budget in BUDGETS:
        sampler = Sampler()
        sampler.start()
        started = time.perf_counter()
        response = post(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": budget,
                "temperature": 0,  # greedy, to match the in-process paths
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        latency = time.perf_counter() - started
        peak = sampler.stop()

        tokens = response["usage"]["completion_tokens"]
        runs.append(
            {
                "budget": budget,
                "tokens": tokens,
                "latency_s": latency,
                "tok_per_s": tokens / latency,
                "peak_mib": peak,
                "baseline_mib": baseline,
                "decode_mib": (peak - baseline) if (peak and baseline) else None,
                "note": "server preallocates --gpu-memory-utilization;"
                " per-request delta is not comparable",
            }
        )
        print(
            f"budget={budget:>4} tokens={tokens:>4} {latency:>6.2f}s "
            f"{runs[-1]['tok_per_s']:>6.1f} tok/s  server={peak} MiB"
        )

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "path": PATH,
        "label": LABEL,
        "model": MODEL,
        "runs": runs,
        "meta": {
            "served_models": served,
            "server_pid": pid,
            "decoding": "greedy",
            "fingerprint": response.get("system_fingerprint"),
            "memory_caveat": "cross-process: nvidia-smi sampling, not a torch allocator peak",
            "host": platform.node(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    out_path = RESULTS / f"{PATH}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
