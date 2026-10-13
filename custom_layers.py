from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

__all__ = [
    "n_freq_of",
    "spectral_filter",
    "quantize",
    "seq_codec",
    "quantize_asym",
    "raw_codec",
    "head_codec",
    "SeqFreqSelector",
    "seq_filter_ste",
    "FreqSelector",
    "KVFreqCompressor",
    "FFTCompressedCache",
    "SparseSpectralCache",
]


def n_freq_of(head_dim: int) -> int:
    """Number of rFFT coefficients for a real input of length D (bin 0..D/2)"""
    return head_dim // 2 + 1


def spectral_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """rFFT x along head_dim -> multiply by mask -> irFFT

    x:    (..., D)   real. fft does not support bf16, so cast to float32 internally
    mask: (..., F)   F = D//2+1. assumed in [0, 1], not enforced
    out:  (..., D)   same dtype as x
    """
    head_dim = x.shape[-1]
    spec = torch.fft.rfft(x.float(), dim=-1)
    spec = spec * mask.float()  # complex * real -> same scaling on real/imag parts
    return torch.fft.irfft(spec, n=head_dim, dim=-1).to(x.dtype)


# ------------------------------------------------------------ sequence-axis codec
# Storage format (one block, one head, one channel = block real values):
#   coefs  : real/imag parts of the keep selected rFFT coefficients. bf16 as is, or int8/int4 + one bf16 scale
#   index  : selected frequency positions. The smaller of an F-bit bitmask and an index list (uint8 x keep). lowpass is fixed, so not needed
# The functions below reconstruct from the values in this format alone and count the stored tensors' bytes directly
_QMAX = {"int8": 127, "int4": 7, "int2": 1}
_BITS = {"bf16": 16, "int8": 8, "int4": 4, "int2": 2, "int1": 1}  # int1 is for asymmetric quantization only


def quantize(v: torch.Tensor, coef: str, dim) -> tuple[torch.Tensor, int]:
    """Symmetric absmax quantization -> reconstructed values and stored bytes. Elements grouped by `dim` share one scale

    scale is bf16 (2 bytes). int4 counts two values per byte, int2 (-1/0/+1) four
    """
    if coef == "bf16":
        return v.to(torch.bfloat16).float(), v.numel() * 2
    qmax = _QMAX[coef]
    scale = v.abs().amax(dim=dim, keepdim=True).to(torch.bfloat16).float().clamp_min(1e-8)
    q = (v / scale * qmax).round().clamp(-qmax, qmax)
    return q * scale / qmax, v.numel() * _BITS[coef] // 8 + scale.numel() * 2


def seq_codec(
    x: torch.Tensor,
    keep_ratio: float,
    block: int = 64,
    mode: str = "energy",
    coef: str = "bf16",
    bias: torch.Tensor | None = None,
    coef_group: int | None = None,
    coef_asym: bool = False,
) -> tuple[torch.Tensor, int]:
    """Store x (B, H, S, D) as sequence-axis block spectra, then reconstruct

    mode: lowpass | energy | random | learned (= top keep by log|X| + bias)
    coef_group: number of coefficients sharing a scale in int8/int4 (grouped in frequency order). None means
                one scale per (block, channel). DC is tens of times larger than the other coefficients, so
                a single shared scale wipes small coefficients to 0
    coef_asym: quantize coefficients with asymmetric min-max (`quantize_asym`). Stores min/max instead of scale
    Returns: (reconstructed values, stored bytes). Tail tokens of a partially filled block count at original bf16 size
    """
    B, H, S, D = x.shape
    n_blk = S // block
    tail = x[..., n_blk * block :, :]
    nbytes = tail.numel() * tail.element_size()
    if n_blk == 0:
        return x, nbytes
    n_freq = block // 2 + 1
    keep = max(1, min(n_freq, round(keep_ratio * n_freq)))
    body = x[..., : n_blk * block, :].float().reshape(B, H, n_blk, block, D)
    spec = torch.fft.rfft(body, dim=-2)  # (B, H, n_blk, F, D)

    if mode == "lowpass":
        idx = torch.arange(keep, device=x.device).view(1, 1, 1, keep, 1).expand(B, H, n_blk, keep, D)
    else:
        if mode == "energy":
            scores = spec.abs()
        elif mode == "learned":
            scores = spec.abs().clamp_min(1e-6).log() + bias[:, None]  # bias: (H, F, D)
        elif mode == "random":
            scores = torch.rand(spec.shape, device=x.device)
        else:
            raise ValueError(f"unknown mode: {mode}")
        idx = scores.topk(keep, dim=-2).indices
        index_bytes = min((n_freq + 7) // 8, keep * (1 if n_freq <= 256 else 2))  # bitmask vs index list
        nbytes += B * H * n_blk * D * index_bytes

    idx = idx.sort(dim=-2).values  # sort by frequency (coefficients of similar magnitude share a scale)
    picked = spec.gather(-2, idx)
    parts = torch.stack([picked.real, picked.imag], dim=-2)  # (B, H, n, keep, 2, D)
    qfn = quantize_asym if coef_asym else quantize
    if coef_group is None or coef == "bf16" or coef_group >= keep:
        parts, coef_bytes = qfn(parts, coef, dim=(-3, -2))  # scale: one per (block, channel)
    else:
        pad = -keep % coef_group
        grouped = torch.nn.functional.pad(parts, (0, 0, 0, 0, 0, pad))
        grouped = grouped.reshape(B, H, n_blk, -1, coef_group, 2, D)
        grouped, coef_bytes = qfn(grouped, coef, dim=(-3, -2))
        coef_bytes -= B * H * n_blk * D * 2 * pad * _BITS[coef] // 8  # padding is not stored
        parts = grouped.reshape(B, H, n_blk, keep + pad, 2, D)[..., :keep, :, :]
    nbytes += coef_bytes
    spec_hat = torch.zeros_like(spec).scatter(-2, idx, torch.complex(parts[..., 0, :], parts[..., 1, :]))
    out = torch.fft.irfft(spec_hat, n=block, dim=-2).reshape(B, H, n_blk * block, D)
    return torch.cat([out.to(x.dtype), tail], dim=-2), nbytes


def quantize_asym(v: torch.Tensor, coef: str, dim) -> tuple[torch.Tensor, int]:
    """Asymmetric min-max quantization (uses all 2^bits levels). Stores min/max per group as bf16"""
    if coef == "bf16":
        return v.to(torch.bfloat16).float(), v.numel() * 2
    levels = 2 ** _BITS[coef] - 1
    lo = v.amin(dim=dim, keepdim=True).to(torch.bfloat16).float()
    hi = v.amax(dim=dim, keepdim=True).to(torch.bfloat16).float()
    step = ((hi - lo) / levels).clamp_min(1e-8)
    q = ((v - lo) / step).round().clamp(0, levels)
    return q * step + lo, v.numel() * _BITS[coef] // 8 + lo.numel() * 4


def raw_codec(x: torch.Tensor, which: str, coef: str = "int8", group: int = 64) -> tuple[torch.Tensor, int]:
    """Baseline: quantize KV directly without FFT (KIVI style, asymmetric min-max)

    K stores min/max per channel over `group` tokens, V per token over `group` channels
    """
    if coef == "bf16":
        return x, x.numel() * x.element_size()
    B, H, S, D = x.shape
    if which == "k":
        n_grp = S // group
        body = x[..., : n_grp * group, :].float().reshape(B, H, n_grp, group, D)
        deq, nbytes = quantize_asym(body, coef, dim=-2)
        deq = deq.reshape(B, H, n_grp * group, D)
        tail = x[..., n_grp * group :, :]
        nbytes += tail.numel() * tail.element_size()
        return torch.cat([deq.to(x.dtype), tail], dim=-2), nbytes
    g = min(group, D)
    deq, nbytes = quantize_asym(x.float().reshape(B, H, S, D // g, g), coef, dim=-1)
    return deq.reshape(B, H, S, D).to(x.dtype), nbytes


def head_codec(x: torch.Tensor, keep_ratio: float, coef: str = "bf16") -> tuple[torch.Tensor, int]:
    """For comparison: head_dim-axis energy top-k (method of Sections 2-8). bitmask + coefficients per token and head"""
    D = x.shape[-1]
    n_freq = D // 2 + 1
    keep = max(1, min(n_freq, round(keep_ratio * n_freq)))
    spec = torch.fft.rfft(x.float(), dim=-1)
    idx = spec.abs().topk(keep, dim=-1).indices
    picked = spec.gather(-1, idx)
    parts, nbytes = quantize(torch.stack([picked.real, picked.imag], dim=-1), coef, dim=(-2, -1))
    nbytes += x[..., 0].numel() * ((n_freq + 7) // 8)
    spec_hat = torch.zeros_like(spec).scatter(-1, idx, torch.complex(parts[..., 0], parts[..., 1]))
    return torch.fft.irfft(spec_hat, n=D, dim=-1).to(x.dtype), nbytes


class SeqFreqSelector(nn.Module):
    """Learned sequence-axis coefficient selection. score = log|X| + bias[layer, k|v][head, freq, channel]

    - bias initialized to 0 → training starts at the same point as energy top-k; gain over energy = training effect
    - Keeps the magnitude information (|X|); bias only corrects for frequencies/channels that matter downstream
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, block: int) -> None:
        super().__init__()
        self.block = block
        n_freq = block // 2 + 1
        self.bias = nn.ParameterDict({
            f"{l}_{w}": nn.Parameter(torch.zeros(num_heads, n_freq, head_dim))
            for l in range(num_layers) for w in ("k", "v")
        })

    def get(self, layer_idx: int, which: str) -> torch.Tensor:
        return self.bias[f"{layer_idx}_{which}"]


def seq_filter_ste(
    x: torch.Tensor, keep_ratio: float, block: int, bias: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """For training. forward uses the learned top-k hard mask, backward the gradient of the sigmoid soft mask (STE)"""
    B, H, S, D = x.shape
    n_blk = S // block
    n_freq = block // 2 + 1
    keep = max(1, min(n_freq, round(keep_ratio * n_freq)))
    body = x[..., : n_blk * block, :].float().reshape(B, H, n_blk, block, D)
    spec = torch.fft.rfft(body, dim=-2)
    scores = spec.abs().clamp_min(1e-6).log() + bias[:, None]
    top = scores.topk(min(keep + 1, n_freq), dim=-2)
    hard = torch.zeros_like(scores).scatter(-2, top.indices[..., :keep, :], 1.0)
    if keep < n_freq:  # boundary = midpoint between the keep-th and (keep+1)-th scores
        thr = top.values[..., keep - 1 : keep + 1, :].mean(-2, keepdim=True).detach()
    else:
        thr = top.values[..., -1:, :].detach() - 1.0
    soft = torch.sigmoid((scores - thr) / tau)
    mask = hard + soft - soft.detach()
    out = torch.fft.irfft(spec * mask, n=block, dim=-2).reshape(B, H, n_blk * block, D)
    return torch.cat([out.to(x.dtype), x[..., n_blk * block :, :]], dim=-2)


class FreqSelector(nn.Module):
    """MLP that computes per-frequency importance from the head vector of one token

    - The last Linear is initialized with weight 0, bias `init_logit`
    - At the start of training the mask is sigmoid(init_logit) ~= 1 at every frequency → student output nearly identical to the teacher
    - The sparsity loss then lowers the mask gradually, so there is no quality drop right after initialization
    """

    def __init__(
        self,
        head_dim: int,
        hidden_dim: int | None = None,
        init_logit: float = 4.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_freq = n_freq_of(head_dim)
        self.temperature = temperature
        hidden_dim = hidden_dim or 2 * head_dim
        self.net = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_freq),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_logit)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Logit before the mask. The Top-K selection criterion"""
        return self.net(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scores(x) / self.temperature)


class KVFreqCompressor(nn.Module):
    """Container of per-layer Selector_K / Selector_V

    - K and V play different roles, so their selectors are separate (`share_kv=True` shares them)
    - Changing only `mode` runs both the Stage 1 (soft) and Stage 2 (hard top-k, static low-pass,
      random) experiments with the same selector

    mode:
        "off"       no compression (= teacher)
        "soft"      sigmoid mask. Used during training
        "topk"      1 for the top keep_ratio by selector score, 0 elsewhere (adaptive per token and head)
        "threshold" 1 only for frequencies with sigmoid mask > threshold
        "lowpass"   1 for the lowest keep_ratio of frequencies (untrained static baseline)
        "energy"    1 for the top keep_ratio by |coef| (untrained adaptive upper bound)
        "random"    1 for a random keep_ratio (control)

    - `targets` restricts compression to K or V only, `active_layers` to a subset of layers. For finding sensitive spots
    - `renorm=True`: restores the norm of the reconstructed vector to the original norm. Dropping frequencies also
      reduces energy; this option checks separately whether the magnitude change alone alters the output
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        hidden_dim: int | None = None,
        share_layers: bool = False,
        share_kv: bool = False,
        init_logit: float = 4.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_freq = n_freq_of(head_dim)
        self.share_layers = share_layers
        self.share_kv = share_kv

        def bank() -> nn.ModuleList:
            count = 1 if share_layers else num_layers
            return nn.ModuleList(
                FreqSelector(head_dim, hidden_dim, init_logit, temperature) for _ in range(count)
            )

        self.selectors_k = bank()
        self.selectors_v = bank() if not share_kv else self.selectors_k

        # Runtime settings. Read from inside the cache, so plain attributes rather than parameters
        self.mode = "soft"
        self.keep_ratio = 1.0
        self.threshold = 0.5
        self.straight_through = False  # whether to pass the soft gradient through the hard mask
        self.renorm = False
        self.targets = {"k", "v"}
        self.active_layers = set(range(num_layers))
        self.collect = True
        self._log: list[dict] = []

    # ------------------------------------------------------------------ mask
    def selector(self, which: str, layer_idx: int) -> FreqSelector:
        bank = self.selectors_k if which == "k" else self.selectors_v
        return bank[0 if self.share_layers else layer_idx]

    def _hard_from_scores(self, scores: torch.Tensor, keep: int) -> torch.Tensor:
        """0/1 mask that is 1 only at the top keep frequencies"""
        hard = torch.zeros_like(scores)
        idx = scores.topk(keep, dim=-1).indices
        return hard.scatter(-1, idx, 1.0)

    def make_mask(
        self, x: torch.Tensor, which: str, layer_idx: int, spec: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (..., D) -> mask: (..., F). `spec` is the rFFT result, used only in energy mode"""
        keep = max(1, min(self.n_freq, round(self.keep_ratio * self.n_freq)))

        if self.mode == "energy":
            if spec is None:
                spec = torch.fft.rfft(x.float(), dim=-1)
            return self._hard_from_scores(spec.abs(), keep)
        if self.mode == "lowpass":
            mask = torch.zeros(*x.shape[:-1], self.n_freq, device=x.device, dtype=torch.float32)
            mask[..., :keep] = 1.0
            return mask
        if self.mode == "random":
            noise = torch.rand(*x.shape[:-1], self.n_freq, device=x.device)
            return self._hard_from_scores(noise, keep)

        selector = self.selector(which, layer_idx)
        scores = selector.scores(x)
        soft = torch.sigmoid(scores / selector.temperature)
        if self.mode == "soft":
            return soft
        if self.mode == "threshold":
            hard = (soft > self.threshold).float()
        elif self.mode == "topk":
            hard = self._hard_from_scores(scores, keep)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.straight_through:
            hard = hard + soft - soft.detach()  # forward is hard, backward is soft
        return hard

    # --------------------------------------------------------------- compress
    def compress(
        self, x: torch.Tensor, which: str, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, H, S, D) -> reconstruction of the same shape + mask (B, H, S, F)"""
        if self.mode == "off" or layer_idx not in self.active_layers or which not in self.targets:
            return x, None
        mask = self.make_mask(x, which, layer_idx)
        out = spectral_filter(x, mask)
        if self.renorm:
            scale = x.float().norm(dim=-1, keepdim=True) / out.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out = (out.float() * scale).to(x.dtype)
        if self.collect:
            self._log.append(
                {
                    "layer": layer_idx,
                    "which": which,
                    "mask": mask,  # needed for the loss, so the graph is kept
                }
            )
        return out, mask

    # ------------------------------------------------------------------ stats
    def reset_log(self) -> None:
        self._log = []

    @property
    def log(self) -> list[dict]:
        return self._log

    def mask_mean(self) -> torch.Tensor:
        """Overall mean usage for the sparsity loss. Stays attached to the graph"""
        if not self._log:
            raise RuntimeError("no log. run forward first")
        return torch.stack([e["mask"].mean() for e in self._log]).mean()

    @torch.no_grad()
    def usage_table(self) -> dict[tuple[int, str], float]:
        """(layer, 'k'|'v') -> mean mask value"""
        return {(e["layer"], e["which"]): float(e["mask"].mean()) for e in self._log}

    @torch.no_grad()
    def per_token_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """Per-token usage of the given layer (B, H, S) -> (S,) mean"""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=-1).mean(dim=(0, 1)).float().cpu()
        raise KeyError((layer_idx, which))

    @torch.no_grad()
    def per_freq_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """Per-frequency mean usage of the given layer (F,)"""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=(0, 1, 2)).float().cpu()
        raise KeyError((layer_idx, which))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class FFTCompressedCache(DynamicCache):
    """Behaves like DynamicCache, but passes KV through the spectral filter right before storing

    For Stage 1/2. Stored tensors keep the original shape (reconstructed D-dim real vectors), so only the
    information content drops; bytes are unchanged. Byte savings are handled by `SparseSpectralCache`
    """

    def __init__(self, compressor: KVFreqCompressor | None = None, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.compressor = compressor

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.compressor is not None:
            key_states, _ = self.compressor.compress(key_states, "k", layer_idx)
            value_states, _ = self.compressor.compress(value_states, "v", layer_idx)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


class SparseSpectralCache(DynamicCache):
    """Stage 3: cache that stores only (frequency index, complex coefficient) instead of the full spectrum

    Picks `keep` coefficients per token and head and stores them in the format below

        idx  : int16      (B, H, S, keep)
        coef : real/imag parts, each `coef_dtype` (B, H, S, keep)

    - On every hand-off to attention: scatter into a zero-filled spectrum, then reconstruct via irFFT
    - For checking that store-compressed, reconstruct-on-read works
    - `footprint_bytes()`: returns the bytes actually stored right now
    """

    def __init__(
        self,
        compressor: KVFreqCompressor,
        keep: int,
        config=None,
        coef_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.compressor = compressor
        self.keep = keep
        self.coef_dtype = coef_dtype
        self.head_dim = compressor.head_dim
        self.n_freq = compressor.n_freq
        # layer_idx -> {'k'|'v': {'idx', 're', 'im'}}
        self.store: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    # ------------------------------------------------------------ internal helpers
    def _encode(self, x: torch.Tensor, which: str, layer_idx: int) -> dict[str, torch.Tensor]:
        scores = self.compressor.selector(which, layer_idx).scores(x)
        idx = scores.topk(self.keep, dim=-1).indices  # (B, H, S, keep)
        spec = torch.fft.rfft(x.float(), dim=-1)
        picked = spec.gather(-1, idx)
        return {
            "idx": idx.to(torch.int16),
            "re": picked.real.to(self.coef_dtype),
            "im": picked.imag.to(self.coef_dtype),
        }

    def _decode(self, packed: dict[str, torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
        idx = packed["idx"].long()
        coef = torch.complex(packed["re"].float(), packed["im"].float())
        spec = torch.zeros(*idx.shape[:-1], self.n_freq, dtype=coef.dtype, device=idx.device)
        spec.scatter_(-1, idx, coef)
        return torch.fft.irfft(spec, n=self.head_dim, dim=-1).to(out_dtype)

    @staticmethod
    def _append(slot: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> None:
        for key in ("idx", "re", "im"):
            slot[key] = torch.cat([slot[key], new[key]], dim=-2)

    # ------------------------------------------------------------------ cache
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        dtype = key_states.dtype
        bucket = self.store.setdefault(layer_idx, {})
        for which, states in (("k", key_states), ("v", value_states)):
            packed = self._encode(states, which, layer_idx)
            if which in bucket:
                self._append(bucket[which], packed)
            else:
                bucket[which] = packed

        # Length/mask bookkeeping is delegated to the parent (DynamicCache). The parent only receives a
        # head_dim-1 dummy; the real KV is kept solely as the sparse representation in self.store
        stub = key_states.new_zeros((*key_states.shape[:-1], 1))
        super().update(stub, stub, layer_idx, *args, **kwargs)

        keys = self._decode(bucket["k"], dtype)
        values = self._decode(bucket["v"], dtype)
        return keys, values

    def footprint_bytes(self) -> int:
        """Number of bytes actually stored right now"""
        total = 0
        for bucket in self.store.values():
            for packed in bucket.values():
                for tensor in packed.values():
                    total += tensor.numel() * tensor.element_size()
        return total
