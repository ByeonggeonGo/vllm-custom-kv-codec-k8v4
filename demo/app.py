#!/usr/bin/env python
"""KV 캐시 코덱 비교 웹 데모 (표준 라이브러리만 사용)

    demo/servers.sh up        # vLLM 서버 5개
    python demo/app.py        # http://localhost:8200

- 같은 질문을 선택한 서버들에 동시에 보내 답변을 SSE로 중계
- 1초마다 서버별 GPU 프로세스 메모리(nvidia-smi)와 KV 캐시 사용률(vLLM /metrics) 수집
- 요청별 TTFT, 생성 속도, 토큰 수, KV 캐시 점유 바이트 기록
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PORT = int(os.environ.get("PORT", 8200))
POOL_BYTES = int(os.environ.get("POOL_MB", 256)) * 1024 * 1024
CORPUS_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
CORPUS = REPO / "results" / "corpus_gutenberg.txt"

# name은 servers.sh의 --served-model-name, 컨테이너는 kvdemo-<name>
SERVERS = [
    {"name": "bf16", "label": "bf16 (no compression)", "port": 8101, "ours": False,
     "note": "vLLM default KV cache"},
    {"name": "vllm-int8", "label": "vLLM int8", "port": 8102, "ours": False,
     "note": "built-in, per (token, head) scale"},
    {"name": "vllm-int4", "label": "vLLM int4", "port": 8103, "ours": False,
     "note": "built-in, rotation + zero-point"},
    {"name": "ours-k8v4", "label": "ours K8 V4", "port": 8104, "ours": True,
     "note": "plugin: K 8-bit, V 4-bit, rotation"},
    {"name": "ours-k4v4", "label": "ours K4 V4", "port": 8105, "ours": True,
     "note": "plugin: 4-bit, channel-mean removal + rotation"},
]
BY_NAME = {s["name"]: s for s in SERVERS}

HISTORY: list[dict] = []   # 1초 간격 표본 {t, servers: {name: {gpu_mib, kv_usage}}}
RUNS: list[dict] = []      # 요청 결과
STATIC: dict[str, dict] = {}  # 서버별 고정 정보 (수용 토큰 수 등)
LOCK = threading.Lock()


# ---------------------------------------------------------------------- 수집
def _get(url: str, timeout: float = 1.5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def _metrics(port: int) -> dict | None:
    """vLLM /metrics에서 KV 캐시 사용률과 블록 설정 추출"""
    text = _get(f"http://localhost:{port}/metrics")
    if text is None:
        return None
    out: dict = {}
    m = re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)", text, re.M)
    if m:
        out["kv_usage"] = float(m.group(1))
    m = re.search(r"^vllm:cache_config_info\{([^}]*)\}", text, re.M)
    if m:
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        out["block_size"] = int(labels.get("block_size") or 16)
        out["num_gpu_blocks"] = int(labels.get("num_gpu_blocks") or 0)
    return out


def _gpu_by_pid() -> dict[int, int]:
    try:
        raw = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return {}
    out = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = int(parts[1])
    return out


def _container_pids(name: str) -> set[int]:
    try:
        raw = subprocess.run(["docker", "top", f"kvdemo-{name}", "-eo", "pid"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return set()
    return {int(x) for x in raw.split() if x.isdigit()}


def collector() -> None:
    pids: dict[str, set[int]] = {}
    tick = 0
    while True:
        if tick % 15 == 0:
            pids = {s["name"]: _container_pids(s["name"]) for s in SERVERS}
        gpu = _gpu_by_pid()
        sample = {"t": time.time(), "servers": {}}
        for s in SERVERS:
            m = _metrics(s["port"])
            if m is None:
                STATIC.pop(s["name"], None)
                continue
            if "num_gpu_blocks" in m and m["num_gpu_blocks"]:
                capacity = m["num_gpu_blocks"] * m["block_size"]
                STATIC[s["name"]] = {"capacity_tokens": capacity, "bytes_per_token": POOL_BYTES / capacity}
            sample["servers"][s["name"]] = {
                "gpu_mib": sum(gpu.get(p, 0) for p in pids.get(s["name"], ())),
                "kv_usage": m.get("kv_usage", 0.0),
            }
        with LOCK:
            HISTORY.append(sample)
            del HISTORY[:-900]
        tick += 1
        time.sleep(1.0)


# ---------------------------------------------------------------------- 지문
_corpus_text: str | None = None


def passage(n_tokens: int) -> str:
    """문맥 길이를 늘리는 데 쓰는 소설 본문. 토큰 수는 글자 수로 근사 (영문 약 4.4자/토큰)"""
    global _corpus_text
    if _corpus_text is None:
        if not CORPUS.exists():
            CORPUS.parent.mkdir(exist_ok=True)
            CORPUS.write_text(urllib.request.urlopen(CORPUS_URL, timeout=30).read().decode("utf-8", "ignore"))
        _corpus_text = CORPUS.read_text(encoding="utf-8", errors="ignore")
    start = 20000
    return _corpus_text[start : start + int(n_tokens * 4.4)]


# ---------------------------------------------------------------------- 생성
def run_one(name: str, prompt: str, max_tokens: int, run_id: str, out: queue.Queue) -> None:
    """서버 하나에 스트리밍 요청. 토큰과 최종 통계를 큐에 넣음"""
    server = BY_NAME[name]
    body = json.dumps({
        "model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://localhost:{server['port']}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage, peak, last_poll, text = {}, 0.0, 0.0, []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        text.append(piece)
                        out.put({"type": "token", "server": name, "text": piece})
                        if now - last_poll > 0.25:  # 요청 중 KV 캐시 사용률 최댓값
                            last_poll = now
                            m = _metrics(server["port"])
                            if m:
                                peak = max(peak, m.get("kv_usage", 0.0))
    except Exception as e:  # 연결 실패, 문맥 초과 등
        out.put({"type": "error", "server": name, "message": str(e)})
        out.put({"type": "done", "server": name})
        return

    total = time.perf_counter() - t0
    n_out = usage.get("completion_tokens", len(text))
    n_in = usage.get("prompt_tokens", 0)
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    static = STATIC.get(name, {})
    result = {
        "type": "result", "server": name, "run_id": run_id, "t": time.time(),
        "prompt_tokens": n_in, "completion_tokens": n_out,
        "ttft_ms": (t_first - t0) * 1000 if t_first else None,
        "total_ms": total * 1000,
        "tok_per_s": (n_out - 1) / decode_s if (decode_s and n_out > 1) else None,
        "kv_mb": (n_in + n_out) * static.get("bytes_per_token", 0) / 2**20,
        "kv_usage_peak": peak,
        "text": "".join(text),
    }
    with LOCK:
        RUNS.append(result)
        del RUNS[:-500]
    out.put(result)
    out.put({"type": "done", "server": name})


# ---------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # 조용히
        pass

    def _json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        if path == "/":
            data = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/state":
            since = float(params.get("since", 0))
            with LOCK:
                hist = [h for h in HISTORY if h["t"] > since]
                runs = [r for r in RUNS if r["t"] > since]
            servers = [{**s, **STATIC.get(s["name"], {}), "up": s["name"] in STATIC} for s in SERVERS]
            self._json({"servers": servers, "history": hist, "runs": runs, "pool_mb": POOL_BYTES / 2**20,
                        "now": time.time()})
        elif path == "/api/passage":
            self._json({"text": passage(int(params.get("tokens", 0)))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        names = [n for n in body.get("servers", []) if n in BY_NAME]
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 128))
        run_id = body.get("run_id") or f"{time.time():.3f}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        out: queue.Queue = queue.Queue()
        if body.get("sequential", True):  # GPU 한 장을 나눠 쓰므로 시간 측정은 한 서버씩
            def chain() -> None:
                for name in names:
                    run_one(name, prompt, max_tokens, run_id, out)
            threading.Thread(target=chain, daemon=True).start()
        else:
            for name in names:
                threading.Thread(target=run_one, args=(name, prompt, max_tokens, run_id, out), daemon=True).start()
        pending = len(names)
        try:
            while pending:
                event = out.get()
                if event["type"] == "done":
                    pending -= 1
                    continue
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: {\"type\": \"end\"}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    threading.Thread(target=collector, daemon=True).start()
    print(f"http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
