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


# --------------------------------------------------------------------- basics
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_id: str = "Qwen/Qwen3-0.6B", device: str = "cuda", dtype=torch.bfloat16):
    """Load the model, then **freeze all parameters**. Only the selector is trained"""
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


# --------------------------------------------------------------------- data
_SOURCES = {
    # Project Gutenberg, Pride and Prejudice. Plain English prose
    "gutenberg": ("url", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    # Plain-text txt on the HF hub. Downloadable without the datasets package
    "tinystories": ("hf", ("roneneldan/TinyStories", "TinyStories-valid.txt")),
}


def load_text(source: str = "gutenberg") -> str:
    """Return the corpus as a string. Given a file path, read that file"""
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
    """Split a continuous token stream into non-overlapping (B, S) chunks

    The leading `skip_chars` are excluded as non-body text (Gutenberg header etc.)
    """
    ids = tok(text[skip_chars:], return_tensors="pt").input_ids[0]
    need = seq_len * batch_size * n_batches
    if ids.numel() < need:
        raise ValueError(f"not enough tokens: {ids.numel()} < {need}. reduce seq_len or n_batches")
    ids = ids[:need].view(n_batches, batch_size, seq_len)
    order = torch.randperm(n_batches, generator=torch.Generator().manual_seed(seed))
    return [ids[i].to(device) for i in order]


# --------------------------------------------------------------------- forward
def forward_logits(model, ids: torch.Tensor, cache=None) -> torch.Tensor:
    """(B, S) -> (B, S, V). cache=None is the uncompressed original path"""
    out = model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
    return out.logits


@torch.no_grad()
def teacher_logits(model, ids: torch.Tensor) -> torch.Tensor:
    return forward_logits(model, ids, cache=None).float()


def distill_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Token-mean KL(teacher || student). Assumes the teacher is detached"""
    t_logp = F.log_softmax(teacher.float(), dim=-1)
    s_logp = F.log_softmax(student.float(), dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / teacher.shape[1]


def token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Next-token NLL (nats/token). perplexity = exp(this value)"""
    pred = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    target = ids[:, 1:].reshape(-1)
    return F.cross_entropy(pred, target)


# ---------------------------------------------------------------------- eval
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
    """KL vs teacher / top-1 agreement / perplexity / mean mask usage"""
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


# ------------------------------------------------------------------ hypothesis check
class _SpectrumProbe(DynamicCache):
    """Cache that does not compress; only collects per-layer rFFT magnitudes of K/V

    axis="head": FFT the D-dim vector per token and head. One sample = (b, h, s), F = D//2+1
    axis="seq":  FFT the length-S time series per head and channel. One sample = (b, h, d), F = S//2+1
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
    """Statistics for checking whether K/V are sparse in the FFT basis

    axis: "head" = FFT along head_dim, "seq" = along the sequence axis (`_SpectrumProbe`)
    drop_dc: compute after excluding bin 0. The seq-axis DC is the per-channel time mean, so for K with a
             large channel bias the energy piles into DC alone and looks sparser than it is; this avoids that

    Returns:
        mean_mag[(layer, which)]   mean |coef| per frequency (F,)
        energy_curve[(layer, which)] fraction of energy kept when retaining the top m
                                     by magnitude (F,). The faster it approaches 1, the sparser
        n_freq                     F. With axis="seq" it varies with the ids length
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
    """Compress one layer at a time and measure KL. For finding sensitive layers"""
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


# ---------------------------------------------------------------------- training
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
    """Update the selector only. Model weights stay frozen

    L = KL(teacher || student) + lambda_sparse * mask.mean()
                               + lambda_budget * relu(mask.mean() - target_ratio)^2

    - Stage 1: `mode="soft"` + sparsity/budget terms to learn usage
    - Stage 2: `mode="topk", keep_ratio=r` fixes the ratio and learns only what to select. The mask is 0/1,
      so the sparsity term is constant; usually `lambda_sparse=0, target_ratio=None`
    - `straight_through=True`: passes the soft gradient through the hard mask, so top-k selection is trainable too
    - `target_ratio=None`: disables the budget term
    - sparsity/budget coefficients ramp linearly from 0 over the first `warmup` steps, because the initial mask
      is near 1 and the budget-term gradient is large. Without the ramp the selector collapses on the first step
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
    """Evaluate a compression ratio x selection method grid; returns a list of rows in `plot_tradeoff` input format"""
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


# ------------------------------------------------- sequence-axis compression (prefix scheme)
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(model, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """like: (B, H, S, D) -> cos, sin of positions 0..S-1 (1, 1, S, D), float32"""
    pos = torch.arange(like.shape[-2], device=like.device)[None]
    cos, sin = model.model.rotary_emb(like.float(), pos)
    return cos.float()[:, None], sin.float()[:, None]


def _keep_for(keep_ratio, which: str) -> float:
    """keep_ratio is a single float, or a (k, v) tuple giving K/V separately"""
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
    """Convert the K/V of a filled DynamicCache to the storage format in place, then reconstruct

    mode: "lowpass" | "energy" | "random" | "learned"  sequence axis (`seq_codec`)
          "head"                                       head_dim-axis energy (method of Sections 2-8)
          "raw"                                        quantization only, no FFT (`raw_codec`, group=block)
    keep_ratio: float, or a (k, v) tuple giving K/V separately
    coef: storage format of coefficients (values if raw). "bf16" | "int8" | "int4" | "int2"
    coef_group: number of coefficients sharing one scale in sequence-axis coefficient quantization (`seq_codec`)
    coef_asym: quantize coefficients with asymmetric min-max (`seq_codec`)
    pre_rope: undo K's RoPE rotation, compress, then re-rotate
    train: in learned mode, pass gradients through the STE path (`seq_filter_ste`). Skips byte accounting
    Returns: (stored bytes, original bf16 bytes). Layers / K·V not targeted for compression count at original size
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
    """Fill the cache with the leading prefix tokens -> compress -> forward the trailing tokens. (logits, stored bytes, original bytes)"""
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
    """Past (prefix) is a compressed cache, trailing tokens stay original; predict them and compare with the teacher

    1) fill the cache with ids[:, :prefix]  2) compress and reconstruct with `compress_cache_`
    3) forward ids[:, prefix:] on top of that cache. Metrics are measured on the trailing tokens only
    - The sequence-axis filter mixes later-token information into earlier tokens within a block. A query only
      attends to compressed blocks in its own past, so no future information leaks
    - `bytes` = stored bytes of the compressed span (prefix) / original bf16 bytes. Based on the actual stored tensors
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
    """Control: no compression of the past; keep only the last `window` tokens and discard the rest

    - forward only ids[:, prefix-window:] afresh; compare trailing-token metrics with the teacher (full context)
    - window=0 runs only the trailing tokens with no past = the floor a compression method must beat
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
    """Run `evaluate_prefix` over a mode x ratio grid; returns a list of rows in `plot_tradeoff` input format"""
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
    """Train the sequence-axis selector. As in the prefix scheme, minimize trailing-token KL on top of the compressed past

    - Model weights frozen. Gradient path: trailing-token forward -> compressed K/V -> STE mask -> bias
    - The leading prefill is no_grad, so memory scales only with the trailing-token length
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


# ------------------------------------------------------------------ generation/latency
@torch.no_grad()
def greedy_generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 48,
    make_cache=None,
    chat: bool = True,
) -> dict:
    """Loop for decoding the same prompt with the original cache and a compressed cache, for comparison"""
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


# ------------------------------------------------------------------ memory accounting
def dense_kv_bytes(info: dict, seq_len: int, layers: int | None = None, elem_bytes: int = 2) -> int:
    """KV cache bytes of the current scheme. Includes both K and V"""
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
    """Bytes when storing only (index + complex coefficients)

    One coefficient is two values (real/imag), hence `2 * coef_bytes`. Two ways to store the selected frequency positions

    index   : the chosen frequency numbers as int16, keep of them. Simple to implement but grows with keep
    bitmask : a single F-bit bitmask. ceil(F/8) bytes regardless of keep

    bitmask wins for large keep, index for very small keep. With this overhead, keeping half the
    coefficients still costs more than half the memory
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


# ------------------------------------------------------------------ figures
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
    """compression ratio vs KL / perplexity. Multiple modes overlaid in one figure"""
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
