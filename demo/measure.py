"""Measures the numbers of the README table: perplexity, decode speed and time to first token per server
(standard library only)

    python demo/measure.py bf16:8101 vllm-int8:8102 vllm-int4:8103 ours-k8v4:8104 ours-k4v4:8105

- perplexity: token logprobs of six novel passages (~1,500 tokens each) sent to `/v1/completions` with `echo`
- speed: streamed chat requests with a context of 0 and ~1,600 tokens. Best of 3 after one warm-up
- servers are called one after another so timings do not mix. Short on GPU memory: start them in groups
"""
import json, math, sys, time, urllib.request

CORPUS = open("results/corpus_gutenberg.txt", encoding="utf-8", errors="ignore").read()


def post(port, path, body):
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def ppl(name, port):
    nll, n = 0.0, 0
    for i in range(6):
        text = CORPUS[20000 + i * 40000 :][: int(1500 * 4.4)]
        r = json.load(post(port, "/v1/completions", {"model": name, "prompt": text, "max_tokens": 1,
                                                     "echo": True, "logprobs": 0, "temperature": 0}))
        lp = [x for x in r["choices"][0]["logprobs"]["token_logprobs"][:-1] if x is not None]
        nll -= sum(lp); n += len(lp)
    return math.exp(nll / n), n


def speed(name, port, ctx, max_tokens=200):
    prompt = CORPUS[20000 : 20000 + int(ctx * 4.4)] + "\n\nSummarize the passage above in detail."
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = last = None; usage = {}
    for raw in post(port, "/v1/chat/completions", body):
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        c = json.loads(line[5:])
        usage = c.get("usage") or usage
        if any((ch.get("delta") or {}).get("content") for ch in c.get("choices", [])):
            last = time.perf_counter(); first = first or last
    n = usage["completion_tokens"]
    return (n - 1) / (last - first), (first - t0) * 1000, usage["prompt_tokens"], n


for spec in sys.argv[1:]:
    name, port = spec.split(":")
    p, n = ppl(name, port)
    out = [f"{name:16s} ppl {p:7.2f} ({n} tok)"]
    for ctx in (0, 1600):
        speed(name, port, ctx)  # warm-up
        runs = [speed(name, port, ctx) for _ in range(3)]
        best = max(runs)
        out.append(f"ctx {best[2]:4d}: {best[0]:6.1f} tok/s  ttft {min(r[1] for r in runs):5.0f} ms  out {best[3]}")
    print(" | ".join(out), flush=True)
