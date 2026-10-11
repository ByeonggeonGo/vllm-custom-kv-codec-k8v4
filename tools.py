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

from custom_layers import (
    FFTCompressedCache,
    KVFreqCompressor,
    SeqFreqSelector,
    SparseSpectralCache,
    head_codec,
    raw_codec,
    seq_codec,
    seq_filter_ste,
)

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
    "compress_cache_",
    "evaluate_prefix",
    "evaluate_window",
    "sweep_prefix",
    "train_seq_selector",
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
    """모델 로드 후 **모든 파라미터 freeze**. 학습 대상은 selector뿐"""
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
    # Project Gutenberg, Pride and Prejudice. 일반 영어 산문
    "gutenberg": ("url", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    # HF hub의 평문 txt. datasets 패키지 없이 다운로드 가능
    "tinystories": ("hf", ("roneneldan/TinyStories", "TinyStories-valid.txt")),
}


def load_text(source: str = "gutenberg") -> str:
    """코퍼스를 문자열로 반환. 파일 경로를 주면 해당 파일을 읽음"""
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
    """연속 토큰 스트림을 겹치지 않는 (B, S) 조각으로 분할

    앞부분 `skip_chars`는 Gutenberg 헤더 등 본문 외 영역이라 제외
    """
    ids = tok(text[skip_chars:], return_tensors="pt").input_ids[0]
    need = seq_len * batch_size * n_batches
    if ids.numel() < need:
        raise ValueError(f"토큰 부족: {ids.numel()} < {need}. seq_len 또는 n_batches 축소 필요")
    ids = ids[:need].view(n_batches, batch_size, seq_len)
    order = torch.randperm(n_batches, generator=torch.Generator().manual_seed(seed))
    return [ids[i].to(device) for i in order]


# --------------------------------------------------------------------- forward
def forward_logits(model, ids: torch.Tensor, cache=None) -> torch.Tensor:
    """(B, S) -> (B, S, V). cache가 None이면 압축 없는 원본 경로"""
    out = model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
    return out.logits


@torch.no_grad()
def teacher_logits(model, ids: torch.Tensor) -> torch.Tensor:
    return forward_logits(model, ids, cache=None).float()


def distill_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """토큰 평균 KL(teacher || student). teacher는 detach된 상태 전제"""
    t_logp = F.log_softmax(teacher.float(), dim=-1)
    s_logp = F.log_softmax(student.float(), dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / teacher.shape[1]


def token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """다음 토큰 예측 NLL (nats/token). perplexity = exp(이 값)"""
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
    """teacher 대비 KL / top-1 일치율 / perplexity / 평균 mask 사용률"""
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
    """압축은 하지 않고, 레이어별 K/V의 rFFT 크기만 모으는 캐시

    axis="head": 토큰·head마다 D차원 벡터를 FFT. 샘플 하나 = (b, h, s), F = D//2+1
    axis="seq":  head·채널마다 길이 S 시계열을 FFT. 샘플 하나 = (b, h, d), F = S//2+1
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
    """K/V가 FFT basis에서 sparse한지 확인하는 통계

    axis: "head"면 head_dim 축, "seq"면 sequence 축으로 FFT (`_SpectrumProbe`)
    drop_dc: bin 0 제외 후 계산. seq 축의 DC는 채널별 시간 평균이라, 채널 bias가 큰 K에서는
             DC 하나에 에너지가 몰려 실제보다 sparse하게 보이는 문제 방지

    반환:
        mean_mag[(layer, which)]   주파수별 평균 |계수| (F,)
        energy_curve[(layer, which)] 크기 순 상위 m개를 남겼을 때 보존되는 에너지
                                     비율 (F,). 1에 빨리 가까워질수록 sparse
        n_freq                     F. axis="seq"면 ids 길이에 따라 변동
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
    """한 번에 한 레이어만 압축해 KL 측정. 민감한 레이어 탐색용"""
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
    """selector만 업데이트. 모델 weight는 freeze 상태 유지

    L = KL(teacher || student) + lambda_sparse * mask.mean()
                               + lambda_budget * relu(mask.mean() - target_ratio)^2

    - Stage 1: `mode="soft"` + sparsity/budget 항으로 사용량 학습
    - Stage 2: `mode="topk", keep_ratio=r`로 비율 고정, 선택 대상만 학습. mask가 0/1이라
      sparsity 항이 상수가 되므로 보통 `lambda_sparse=0, target_ratio=None`
    - `straight_through=True`: hard mask에 soft gradient를 전달해 top-k 선택도 학습 가능
    - `target_ratio=None`: budget 항 비활성화
    - sparsity/budget 계수는 처음 `warmup` 스텝 동안 0에서 선형 증가. 초기 mask가 1에
      가까워 budget 항 기울기가 큰 탓. 램프 없이 학습 시 첫 스텝에서 selector 붕괴
      (KL 0.003 → 1.2)
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
    """compression ratio x 선택 방식 격자를 평가해 `plot_tradeoff` 입력 형식의 행 목록으로 반환"""
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


# ------------------------------------------------- sequence 축 압축 (prefix 방식)
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(model, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """like: (B, H, S, D) -> 위치 0..S-1의 cos, sin (1, 1, S, D), float32"""
    pos = torch.arange(like.shape[-2], device=like.device)[None]
    cos, sin = model.model.rotary_emb(like.float(), pos)
    return cos.float()[:, None], sin.float()[:, None]


def _keep_for(keep_ratio, which: str) -> float:
    """keep_ratio는 float 하나, 또는 K/V 따로 주는 (k, v) 튜플"""
    if isinstance(keep_ratio, (tuple, list)):
        return keep_ratio[0] if which == "k" else keep_ratio[1]
    return keep_ratio


def compress_cache_(
    model,
    cache: DynamicCache,
    keep_ratio,
    mode: str = "energy",
    block: int = 64,
    coef: str = "bf16",
    coef_group: int | None = None,
    coef_asym: bool = False,
    targets=("k", "v"),
    skip_layers: int = 0,
    pre_rope: bool = False,
    selector: SeqFreqSelector | None = None,
    train: bool = False,
    tau: float = 1.0,
) -> tuple[int, int]:
    """채워진 DynamicCache의 K/V를 제자리에서 저장 포맷으로 변환 후 복원

    mode: "lowpass" | "energy" | "random" | "learned"  sequence 축 (`seq_codec`)
          "head"                                       head_dim 축 energy (2~8절 방식)
          "raw"                                        FFT 없이 양자화만 (`raw_codec`, group=block)
    keep_ratio: float, 또는 K/V에 따로 줄 (k, v) 튜플
    coef: 계수(raw면 값) 저장 형식. "bf16" | "int8" | "int4" | "int2"
    coef_group: sequence 축 계수 양자화에서 scale을 공유하는 계수 개수 (`seq_codec`)
    coef_asym: 계수 양자화를 비대칭 min-max로 수행 (`seq_codec`)
    pre_rope: K의 RoPE 회전을 풀고 압축한 뒤 재회전
    train: learned 모드에서 STE 경로(`seq_filter_ste`)로 gradient 전달. 바이트 계산 생략
    반환: (저장 바이트, 원본 bf16 바이트). 압축 대상이 아닌 레이어/K·V는 원본 크기로 계산
    """
    stored = original = 0
    for layer_idx, layer in enumerate(cache.layers):
        for which in ("k", "v"):
            x = layer.keys if which == "k" else layer.values
            raw_bytes = x.numel() * x.element_size()
            original += raw_bytes
            if layer_idx < skip_layers or which not in targets:
                stored += raw_bytes
                continue
            keep = _keep_for(keep_ratio, which)
            rope = pre_rope and which == "k"
            if rope:
                cos, sin = _rope_cos_sin(model, x)
                x = (x.float() * cos - _rotate_half(x.float()) * sin).to(x.dtype)

            if mode == "raw":
                y, n = raw_codec(x, which, coef=coef, group=block)
            elif mode == "head":
                y, n = head_codec(x, keep, coef=coef)
            elif train:
                y, n = seq_filter_ste(x, keep, block, selector.get(layer_idx, which), tau), 0
            else:
                bias = selector.get(layer_idx, which) if mode == "learned" else None
                y, n = seq_codec(x, keep, block=block, mode=mode, coef=coef, bias=bias,
                                 coef_group=coef_group, coef_asym=coef_asym)
            stored += n

            if rope:
                y = (y.float() * cos + _rotate_half(y.float()) * sin).to(y.dtype)
            if which == "k":
                layer.keys = y
            else:
                layer.values = y
    return stored, original


def _prefix_student(model, ids, prefix, mode, train=False, **compress_kwargs):
    """앞 prefix 토큰으로 캐시 채움 -> 압축 -> 뒤쪽 토큰 forward. (logits, 저장 바이트, 원본 바이트)"""
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=ids[:, :prefix], past_key_values=cache, use_cache=True)
    if mode == "off":
        stored = original = 1
    else:
        with torch.set_grad_enabled(train):
            stored, original = compress_cache_(model, cache, mode=mode, train=train, **compress_kwargs)
    with torch.set_grad_enabled(train):
        logits = model(input_ids=ids[:, prefix:], past_key_values=cache, use_cache=True).logits
    return logits.float(), stored, original


@torch.no_grad()
def evaluate_prefix(
    model,
    batches: list[torch.Tensor],
    prefix: int,
    keep_ratio=1.0,
    mode: str = "energy",
    **compress_kwargs,
) -> dict:
    """과거(prefix)는 압축 캐시, 뒤쪽 토큰은 원본으로 예측해 teacher와 비교

    1) ids[:, :prefix]로 캐시 채움  2) `compress_cache_`로 압축·복원
    3) ids[:, prefix:]를 그 캐시 위에서 forward. 지표는 뒤쪽 토큰에서만 측정
    - sequence 축 필터는 블록 안 뒤 토큰 정보를 앞 토큰에 섞음. query가 자신보다 과거인
      압축 블록만 참조하므로 미래 정보 누설 없음
    - `bytes` = 압축 구간(prefix)의 저장 바이트 / 원본 bf16 바이트. 실제 저장 텐서 기준
    """
    totals = {"kl": 0.0, "agree": 0.0, "nll": 0.0, "nll_teacher": 0.0}
    stored_sum = original_sum = 0
    for ids in batches:
        t_logits = forward_logits(model, ids)[:, prefix:].float()
        s_logits, stored, original = _prefix_student(model, ids, prefix, mode,
                                                     keep_ratio=keep_ratio, **compress_kwargs)
        tail = ids[:, prefix:]
        totals["kl"] += float(distill_loss(t_logits, s_logits))
        totals["agree"] += float((t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean())
        totals["nll"] += float(token_nll(s_logits, tail))
        totals["nll_teacher"] += float(token_nll(t_logits, tail))
        stored_sum += stored
        original_sum += original
    n = len(batches)
    result = {k: v / n for k, v in totals.items()}
    result["ppl"] = math.exp(result["nll"])
    result["ppl_teacher"] = math.exp(result["nll_teacher"])
    result["bytes"] = stored_sum / original_sum
    result.update(mode=mode, keep_ratio=keep_ratio,
                  **{k: v for k, v in compress_kwargs.items() if k != "selector"})
    return result


@torch.no_grad()
def evaluate_window(model, batches: list[torch.Tensor], prefix: int, window: int) -> dict:
    """대조군: 과거를 압축하지 않고 최근 `window`개 토큰만 남기고 나머지는 폐기

    - ids[:, prefix-window:]만 새로 forward, 뒤쪽 토큰 지표를 teacher(전체 문맥)와 비교
    - window=0이면 과거 없이 뒤쪽 토큰만 실행 = 압축 방식이 넘어야 할 바닥값
    - bytes = window / prefix
    """
    totals = {"kl": 0.0, "agree": 0.0, "nll": 0.0, "nll_teacher": 0.0}
    for ids in batches:
        t_logits = forward_logits(model, ids)[:, prefix:].float()
        s_logits = forward_logits(model, ids[:, prefix - window:])[:, window:].float()
        tail = ids[:, prefix:]
        totals["kl"] += float(distill_loss(t_logits, s_logits))
        totals["agree"] += float((t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean())
        totals["nll"] += float(token_nll(s_logits, tail))
        totals["nll_teacher"] += float(token_nll(t_logits, tail))
    n = len(batches)
    result = {k: v / n for k, v in totals.items()}
    result["ppl"] = math.exp(result["nll"])
    result["ppl_teacher"] = math.exp(result["nll_teacher"])
    result.update(bytes=window / prefix, mode=f"window last {window}", keep_ratio=window / prefix,
                  window=window)
    return result


def sweep_prefix(model, batches, prefix, ratios, modes, label=None, **compress_kwargs) -> list[dict]:
    """`evaluate_prefix`를 mode x ratio 격자로 실행해 `plot_tradeoff` 입력 형식의 행 목록을 반환"""
    rows = []
    for mode in modes:
        for ratio in ratios:
            row = evaluate_prefix(model, batches, prefix, ratio, mode=mode, **compress_kwargs)
            row["mode"] = f"{label} {mode}" if label else mode
            rows.append(row)
            print(f"{row['mode']:>24} keep={str(ratio):<12} bytes={row['bytes']:.3f}x "
                  f"kl={row['kl']:8.4f} agree={row['agree']:.3f} ppl={row['ppl']:10.2f}")
    return rows


def train_seq_selector(
    model,
    selector: SeqFreqSelector,
    batches: list[torch.Tensor],
    prefix: int,
    keep_ratio,
    epochs: int = 1,
    lr: float = 1e-2,
    tau: float = 1.0,
    log_every: int = 20,
    **compress_kwargs,
) -> list[dict]:
    """sequence 축 selector 학습. prefix 방식과 동일하게 압축된 과거 위에서 뒤쪽 토큰의 KL 최소화

    - 모델 weight는 freeze. gradient 경로: 뒤쪽 토큰 forward -> 압축된 K/V -> STE mask -> bias
    - 앞쪽 prefill은 no_grad라 메모리는 뒤쪽 토큰 길이에만 비례
    """
    optim = torch.optim.Adam(selector.parameters(), lr=lr)
    history, step = [], 0
    for epoch in range(epochs):
        for ids in batches:
            with torch.no_grad():
                t_logits = forward_logits(model, ids)[:, prefix:].float()
            s_logits, _, _ = _prefix_student(model, ids, prefix, "learned", train=True,
                                             keep_ratio=keep_ratio, selector=selector, tau=tau,
                                             block=selector.block, **compress_kwargs)
            loss = distill_loss(t_logits, s_logits)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            history.append({"step": step, "epoch": epoch, "kd": float(loss.detach())})
            if step % log_every == 0:
                print(f"step {step:>4}  KL {history[-1]['kd']:.4f}")
            step += 1
    return history


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
    """같은 프롬프트를 원본 캐시와 압축 캐시로 각각 디코딩해 비교하기 위한 루프"""
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
    """현재 방식의 KV cache 바이트. K와 V 모두 포함"""
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
    """(index + complex 계수)만 저장할 때의 바이트

    계수 하나는 실수부/허수부 두 값이라 `2 * coef_bytes`. 선택한 주파수 위치 저장 방식은 두 가지

    index   : 고른 주파수 번호를 int16으로 keep개. 구현은 단순하나 keep에 비례해 증가
    bitmask : F비트 비트마스크 하나. keep과 무관하게 ceil(F/8) 바이트

    keep이 크면 bitmask, 아주 작으면 index가 유리. 이 오버헤드로 계수를 절반 남겨도
    메모리는 절반 이상
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
    """compression ratio 대 KL / perplexity. 여러 mode를 한 그림에 겹쳐 표시"""
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
