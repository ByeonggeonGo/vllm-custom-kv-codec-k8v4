from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from custom_layers import FFTCompressedCache, KVFreqCompressor, SparseSpectralCache

__all__ = [
    "set_seed",
    "load_model",
    "describe_model",
    "load_text",
    "make_batches",
    "forward_logits",
    "teacher_logits",
    "distill_loss",
    "token_nll",
    "evaluate",
    "spectrum_stats",
    "layer_sensitivity",
    "train_selectors",
    "sweep_ratios",
    "greedy_generate",
    "dense_kv_bytes",
    "spectral_kv_bytes",
    "plot_training",
    "plot_tradeoff",
    "plot_layer_usage",
    "plot_freq_profile",
]

CACHE_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------- 기본
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_id: str = "Qwen/Qwen3-0.6B", device: str = "cuda", dtype=torch.bfloat16):
    """모델을 올리고 **모든 파라미터를 freeze** 한다. 학습 대상은 selector뿐이다."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def describe_model(model) -> dict:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {
        "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "q_heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "n_freq": head_dim // 2 + 1,
        "vocab": cfg.vocab_size,
        "dtype": str(model.dtype),
    }


# --------------------------------------------------------------------- 데이터
_SOURCES = {
    # Project Gutenberg, Pride and Prejudice. 일반 영어 산문.
    "gutenberg": ("url", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    # HF hub의 평문 txt. datasets 패키지 없이 받을 수 있다.
    "tinystories": ("hf", ("roneneldan/TinyStories", "TinyStories-valid.txt")),
}


def load_text(source: str = "gutenberg") -> str:
    """코퍼스를 문자열로. 파일 경로를 주면 그 파일을 읽는다."""
    if os.path.exists(source):
        return Path(source).read_text(encoding="utf-8", errors="ignore")

    kind, where = _SOURCES[source]
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / f"corpus_{source}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="ignore")

    if kind == "url":
        import requests

        text = requests.get(where, timeout=60).text
    else:
        from huggingface_hub import hf_hub_download

        repo, name = where
        path = hf_hub_download(repo, name, repo_type="dataset")
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    cached.write_text(text, encoding="utf-8")
    return text


def make_batches(
    tok,
    text: str,
    seq_len: int = 256,
    batch_size: int = 2,
    n_batches: int = 64,
    device: str = "cuda",
    seed: int = 0,
    skip_chars: int = 5000,
) -> list[torch.Tensor]:
    """연속 토큰 스트림을 겹치지 않는 (B, S) 조각으로 자른다.

    앞부분 `skip_chars`는 Gutenberg 헤더 같은 군더더기라 버린다.
    """
    ids = tok(text[skip_chars:], return_tensors="pt").input_ids[0]
    need = seq_len * batch_size * n_batches
    if ids.numel() < need:
        raise ValueError(f"토큰이 모자람: {ids.numel()} < {need}. seq_len이나 n_batches를 줄여라.")
    ids = ids[:need].view(n_batches, batch_size, seq_len)
    order = torch.randperm(n_batches, generator=torch.Generator().manual_seed(seed))
    return [ids[i].to(device) for i in order]


# --------------------------------------------------------------------- forward
def forward_logits(model, ids: torch.Tensor, cache=None) -> torch.Tensor:
    """(B, S) -> (B, S, V). cache가 None이면 압축 없는 원본 경로."""
    out = model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
    return out.logits


@torch.no_grad()
def teacher_logits(model, ids: torch.Tensor) -> torch.Tensor:
    return forward_logits(model, ids, cache=None).float()


def distill_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """토큰 평균 KL(teacher || student). teacher는 detach된 상태를 기대한다."""
    t_logp = F.log_softmax(teacher.float(), dim=-1)
    s_logp = F.log_softmax(student.float(), dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / teacher.shape[1]


def token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """다음 토큰 예측 NLL (nats/token). perplexity = exp(이 값)."""
    pred = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    target = ids[:, 1:].reshape(-1)
    return F.cross_entropy(pred, target)


# ---------------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate(
    model,
    compressor: KVFreqCompressor | None,
    batches: list[torch.Tensor],
    mode: str = "soft",
    keep_ratio: float = 1.0,
    threshold: float = 0.5,
    collect_usage: bool = True,
) -> dict:
    """teacher 대비 KL / top-1 일치율 / perplexity / 평균 mask 사용률."""
    if compressor is not None:
        prev = (compressor.mode, compressor.keep_ratio, compressor.threshold, compressor.collect)
        compressor.mode, compressor.keep_ratio, compressor.threshold = mode, keep_ratio, threshold
        compressor.collect = collect_usage

    totals = {"kl": 0.0, "agree": 0.0, "nll": 0.0, "nll_teacher": 0.0, "usage": 0.0}
    usage_by_layer: dict[tuple[int, str], list[float]] = {}

    for ids in batches:
        t_logits = forward_logits(model, ids, cache=None).float()
        if compressor is None:
            s_logits = t_logits
        else:
            compressor.reset_log()
            cache = FFTCompressedCache(compressor, config=model.config)
            s_logits = forward_logits(model, ids, cache=cache).float()
            for key, value in compressor.usage_table().items():
                usage_by_layer.setdefault(key, []).append(value)
            totals["usage"] += float(compressor.mask_mean()) if compressor.log else 1.0
            del cache
        totals["kl"] += float(distill_loss(t_logits, s_logits))
        totals["agree"] += float((t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean())
        totals["nll"] += float(token_nll(s_logits, ids))
        totals["nll_teacher"] += float(token_nll(t_logits, ids))

    n = len(batches)
    result = {k: v / n for k, v in totals.items()}
    result["ppl"] = math.exp(result["nll"])
    result["ppl_teacher"] = math.exp(result["nll_teacher"])
    result["mode"] = mode
    result["keep_ratio"] = keep_ratio
    result["usage_by_layer"] = {k: sum(v) / len(v) for k, v in usage_by_layer.items()}

    if compressor is not None:
        compressor.mode, compressor.keep_ratio, compressor.threshold, compressor.collect = prev
    return result


# ------------------------------------------------------------------ 가설 점검
class _SpectrumProbe(DynamicCache):
    """압축은 하지 않고, 레이어별 K/V의 rFFT 크기만 모으는 캐시.

    axis="head": 토큰·head마다 D차원 벡터를 FFT. 샘플 하나 = (b, h, s), F = D//2+1.
    axis="seq":  head·채널마다 길이 S 시계열을 FFT. 샘플 하나 = (b, h, d), F = S//2+1.
    """

    def __init__(self, config=None, axis: str = "head"):
        super().__init__(config=config)
        if axis not in ("head", "seq"):
            raise ValueError(f"unknown axis: {axis}")
        self.axis = axis
        self.mag: dict[tuple[int, str], torch.Tensor] = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        for which, states in (("k", key_states), ("v", value_states)):
            if self.axis == "head":
                mag = torch.fft.rfft(states.float(), dim=-1).abs()  # (B, H, S, F)
            else:
                mag = torch.fft.rfft(states.float(), dim=-2).abs()  # (B, H, F, D)
                mag = mag.transpose(-1, -2)                         # (B, H, D, F)
            self.mag[(layer_idx, which)] = mag.reshape(-1, mag.shape[-1]).cpu()
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


@torch.no_grad()
def spectrum_stats(model, ids: torch.Tensor, axis: str = "head", drop_dc: bool = False) -> dict:
    """"K/V가 FFT basis에서 정말 sparse한가"를 재는 가장 직접적인 숫자.

    axis: "head"면 head_dim 축, "seq"면 sequence 축으로 FFT한다 (`_SpectrumProbe`).
    drop_dc: bin 0을 빼고 잰다. seq 축에서는 DC = 채널별 시간 평균이라, 채널 bias가 큰
             K에서는 DC 하나가 에너지를 독차지해 sparse해 보이는 착시를 만든다.

    반환:
        mean_mag[(layer, which)]   주파수별 평균 |계수| (F,)
        energy_curve[(layer, which)] 크기 순 상위 m개를 남겼을 때 보존되는 에너지
                                     비율 (F,). 1에 빨리 붙을수록 sparse하다.
        n_freq                     F. axis="seq"면 ids 길이에 따라 달라진다.
    """
    probe = _SpectrumProbe(config=model.config, axis=axis)
    forward_logits(model, ids, cache=probe)
    mean_mag, energy = {}, {}
    for key, mag in probe.mag.items():
        if drop_dc:
            mag = mag[:, 1:]
        mean_mag[key] = mag.mean(0)
        power = mag.pow(2)
        ordered = power.sort(dim=-1, descending=True).values
        curve = ordered.cumsum(-1) / ordered.sum(-1, keepdim=True).clamp_min(1e-12)
        energy[key] = curve.mean(0)
    n_freq = next(iter(mean_mag.values())).shape[-1]
    return {"mean_mag": mean_mag, "energy_curve": energy, "n_freq": n_freq}


@torch.no_grad()
def layer_sensitivity(
    model,
    compressor: KVFreqCompressor,
    batches: list[torch.Tensor],
    keep_ratio: float = 0.5,
    mode: str = "energy",
    targets=("k", "v"),
) -> list[dict]:
    """한 번에 한 레이어만 압축해 보고 KL을 잰다. 어디가 약한 고리인지 찾는 용도."""
    rows = []
    saved = (compressor.active_layers, compressor.targets)
    for layer in range(compressor.num_layers):
        compressor.active_layers = {layer}
        compressor.targets = set(targets)
        r = evaluate(model, compressor, batches, mode=mode, keep_ratio=keep_ratio,
                     collect_usage=False)
        rows.append({"layer": layer, "kl": r["kl"], "agree": r["agree"], "ppl": r["ppl"]})
    compressor.active_layers, compressor.targets = saved
    return rows


# ---------------------------------------------------------------------- 학습
def train_selectors(
    model,
    compressor: KVFreqCompressor,
    batches: list[torch.Tensor],
    epochs: int = 1,
    lr: float = 1e-3,
    mode: str = "soft",
    keep_ratio: float = 1.0,
    straight_through: bool = True,
    lambda_sparse: float = 0.0,
    target_ratio: float | None = 0.25,
    lambda_budget: float = 10.0,
    warmup: int = 20,
    log_every: int = 10,
    grad_clip: float = 1.0,
) -> list[dict]:
    """selector만 업데이트한다. 모델 weight는 freeze된 상태 그대로.

    L = KL(teacher || student) + lambda_sparse * mask.mean()
                               + lambda_budget * relu(mask.mean() - target_ratio)^2

    Stage 1은 `mode="soft"` + sparsity/budget 항으로 "얼마나 쓸지"를 학습하고,
    Stage 2는 `mode="topk", keep_ratio=r`로 비율을 고정한 채 "무엇을 고를지"만
    학습한다. 후자는 mask가 0/1이라 sparsity 항이 상수가 되므로 보통
    `lambda_sparse=0, target_ratio=None`으로 둔다. `straight_through=True`가
    hard mask에 soft gradient를 흘려 학습이 가능하게 만든다.

    `target_ratio`가 None이면 budget 항은 꺼진다.

    sparsity/budget 계수는 처음 `warmup` 스텝 동안 0에서 선형으로 올린다. 초기 mask는
    1에 가까워 budget 항의 기울기가 크고, 그대로 때리면 첫 스텝에서 selector가
    통째로 망가진다 (KL이 0.003 -> 1.2로 튄다). 램프는 그걸 막는 장치다.
    """
    optim = torch.optim.AdamW(compressor.trainable_parameters(), lr=lr)
    compressor.mode = mode
    compressor.keep_ratio = keep_ratio
    compressor.straight_through = straight_through
    compressor.collect = True
    history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        for ids in batches:
            with torch.no_grad():
                t_logits = forward_logits(model, ids, cache=None).float()

            compressor.reset_log()
            cache = FFTCompressedCache(compressor, config=model.config)
            s_logits = forward_logits(model, ids, cache=cache)

            ramp = min(1.0, (step + 1) / max(1, warmup))
            usage = compressor.mask_mean()
            loss_kd = distill_loss(t_logits, s_logits)
            loss = loss_kd + ramp * lambda_sparse * usage
            if target_ratio is not None:
                loss = loss + ramp * lambda_budget * F.relu(usage - target_ratio).pow(2)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compressor.trainable_parameters(), grad_clip)
            optim.step()

            row = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "kd": float(loss_kd.detach()),
                "usage": float(usage.detach()),
            }
            history.append(row)
            if step % log_every == 0:
                print(
                    f"step {step:>4}  loss {row['loss']:.5f}  KL {row['kd']:.5f}  "
                    f"usage {row['usage']:.3f}"
                )
            step += 1
            del cache, s_logits, t_logits
            compressor.reset_log()
    return history


def sweep_ratios(
    model,
    compressor: KVFreqCompressor,
    batches: list[torch.Tensor],
    ratios=(1.0, 0.875, 0.75, 0.5, 0.25, 0.125, 0.0625),
    modes=("topk", "energy", "lowpass", "random"),
) -> list[dict]:
    """compression ratio x 선택 방식 격자를 평가해 `plot_tradeoff`가 먹는 행들로."""
    rows = []
    for mode in modes:
        for ratio in ratios:
            row = evaluate(model, compressor, batches, mode=mode, keep_ratio=ratio,
                           collect_usage=False)
            row.pop("usage_by_layer", None)
            rows.append(row)
            print(f"{mode:>8} keep={ratio:<7} kl={row['kl']:9.4f} "
                  f"agree={row['agree']:.3f} ppl={row['ppl']:12.2f}")
    return rows


# ------------------------------------------------------------------ 생성/지연
@torch.no_grad()
def greedy_generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 48,
    make_cache=None,
    chat: bool = True,
) -> dict:
    """같은 프롬프트를 원본 캐시와 압축 캐시로 각각 디코딩해 비교하기 위한 루프."""
    if chat:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    enc = tok(prompt, return_tensors="pt").to(model.device)
    cache = make_cache() if make_cache is not None else None
    if cache is None:
        from transformers import DynamicCache

        cache = DynamicCache(config=model.config)

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    cur, out_ids = enc.input_ids, []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = int(out.logits[0, -1].argmax())
        out_ids.append(nxt)
        if nxt in eos_ids:
            break
        cur = torch.tensor([[nxt]], device=model.device, dtype=enc.input_ids.dtype)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    footprint = cache.footprint_bytes() if hasattr(cache, "footprint_bytes") else None
    return {
        "text": tok.decode(out_ids),
        "ids": out_ids,
        "seconds": elapsed,
        "tok_per_s": len(out_ids) / elapsed,
        "footprint_bytes": footprint,
    }


# ------------------------------------------------------------------ 메모리 계산
def dense_kv_bytes(info: dict, seq_len: int, layers: int | None = None, elem_bytes: int = 2) -> int:
    """지금 방식의 KV cache 바이트. K와 V 둘 다 센다."""
    layers = layers or info["layers"]
    return layers * 2 * info["kv_heads"] * info["head_dim"] * seq_len * elem_bytes


def spectral_kv_bytes(
    info: dict,
    seq_len: int,
    keep: int,
    layers: int | None = None,
    coef_bytes: int = 2,
    index_scheme: str = "bitmask",
) -> int:
    """(index + complex 계수)만 저장할 때의 바이트.

    계수 하나는 실수부/허수부 둘이므로 `2 * coef_bytes`가 든다. 어느 주파수를
    골랐는지도 저장해야 하는데 두 가지 방식이 있다:

    index   : 고른 주파수 번호를 int16으로 keep개. 단순하지만 keep에 비례해 비싸다.
    bitmask : F비트짜리 비트마스크 하나. keep과 무관하게 ceil(F/8) 바이트.

    keep이 클수록 bitmask가, 아주 작을 때는 index가 유리하다. 이 오버헤드 때문에
    "계수 절반만 남겼다"가 곧바로 "메모리 절반"이 되지는 않는다.
    """
    layers = layers or info["layers"]
    n_freq = info["n_freq"]
    if index_scheme == "bitmask":
        overhead = (n_freq + 7) // 8
    elif index_scheme == "index":
        overhead = 2 * keep
    else:
        raise ValueError(index_scheme)
    per_vector = keep * 2 * coef_bytes + overhead
    return layers * 2 * info["kv_heads"] * seq_len * per_vector


# ------------------------------------------------------------------ 그림
def plot_training(history: list[dict], ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    steps = [h["step"] for h in history]
    ax[0].plot(steps, [h["kd"] for h in history], color="tab:blue")
    ax[0].set(xlabel="step", ylabel="KL(teacher || student)", title="Distillation loss")
    ax[0].set_yscale("log")
    ax[1].plot(steps, [h["usage"] for h in history], color="tab:orange")
    ax[1].set(xlabel="step", ylabel="mask.mean()", title="mean frequency usage", ylim=(0, 1.02))
    for a in ax:
        a.grid(alpha=0.3)
    return ax


def plot_tradeoff(rows: list[dict], x="keep_ratio", ax=None):
    """compression ratio 대 KL / perplexity. 여러 mode를 한 그림에 겹쳐 그린다."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        sel = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r[x])
        ax[0].plot([r[x] for r in sel], [r["kl"] for r in sel], "o-", label=mode)
        ax[1].plot([r[x] for r in sel], [r["ppl"] for r in sel], "o-", label=mode)
    if rows:
        ax[1].axhline(rows[0]["ppl_teacher"], ls="--", c="gray", label="teacher")
    ax[0].set(xlabel=x, ylabel="KL(teacher || student)", title="output fidelity", yscale="log")
    ax[1].set(xlabel=x, ylabel="perplexity", title="LM quality")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    return ax


def plot_layer_usage(usage_by_layer: dict, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))
    layers = sorted({k[0] for k in usage_by_layer})
    for which, color in (("k", "tab:blue"), ("v", "tab:red")):
        ax.plot(layers, [usage_by_layer.get((l, which), float("nan")) for l in layers],
                "o-", color=color, label=f"selector_{which}")
    ax.set(xlabel="layer", ylabel="mean mask", title="usage per layer", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_freq_profile(compressor: KVFreqCompressor, layers, which="k", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))
    for layer in layers:
        ax.plot(compressor.per_freq_usage(layer, which), label=f"L{layer}")
    ax.set(xlabel="frequency bin (0 = DC)", ylabel="mean mask",
           title=f"usage per frequency bin ({which})", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    return ax
