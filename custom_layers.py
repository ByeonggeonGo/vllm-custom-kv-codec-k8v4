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
    """실수 입력 길이 D에 대한 rFFT 계수 개수 (bin 0..D/2)"""
    return head_dim // 2 + 1


def spectral_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x를 head_dim 축으로 rFFT -> mask 곱 -> irFFT

    x:    (..., D)   실수. bf16은 fft 미지원이라 내부에서 float32로 변환
    mask: (..., F)   F = D//2+1. [0, 1] 범위 전제, 강제하지 않음
    반환:  (..., D)   x와 같은 dtype
    """
    head_dim = x.shape[-1]
    spec = torch.fft.rfft(x.float(), dim=-1)
    spec = spec * mask.float()  # complex * real -> 실수부/허수부에 동일 배율
    return torch.fft.irfft(spec, n=head_dim, dim=-1).to(x.dtype)


# ------------------------------------------------------------ sequence 축 코덱
# 저장 포맷 (블록 하나, head 하나, 채널 하나 = block개 실수):
#   계수   : 고른 keep개 rFFT 계수의 실수부/허수부. bf16 그대로, 또는 int8/int4 + bf16 scale 1개
#   index  : 선택한 주파수 위치. F비트 bitmask와 번호 나열(uint8 x keep) 중 작은 쪽. lowpass는 고정이라 불필요
# 아래 함수는 이 포맷에 담긴 값만으로 복원, 저장 텐서의 바이트를 직접 계산
_QMAX = {"int8": 127, "int4": 7, "int2": 1}
_BITS = {"bf16": 16, "int8": 8, "int4": 4, "int2": 2, "int1": 1}  # int1은 비대칭 양자화 전용


def quantize(v: torch.Tensor, coef: str, dim) -> tuple[torch.Tensor, int]:
    """대칭 absmax 양자화 -> 복원값과 저장 바이트. `dim`으로 묶인 원소끼리 scale 하나 공유

    scale은 bf16(2바이트). int4는 두 값, int2(-1/0/+1)는 네 값을 한 바이트로 계산
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
    """x (B, H, S, D)를 sequence 축 블록 스펙트럼으로 저장 후 복원

    mode: lowpass | energy | random | learned (= log|X| + bias 상위 keep개)
    coef_group: int8/int4에서 scale을 공유하는 계수 개수 (주파수 순서대로 묶음). None이면
                (블록, 채널)당 scale 하나. DC가 다른 계수보다 수십 배 커서, scale 하나로
                묶으면 작은 계수가 0으로 소실
    coef_asym: 계수 양자화를 비대칭 min-max(`quantize_asym`)로 수행. scale 대신 min/max 저장
    반환: (복원값, 저장 바이트). 블록이 덜 찬 꼬리 토큰은 원본 bf16 크기로 계산
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
        index_bytes = min((n_freq + 7) // 8, keep * (1 if n_freq <= 256 else 2))  # bitmask vs 번호 나열
        nbytes += B * H * n_blk * D * index_bytes

    idx = idx.sort(dim=-2).values  # 주파수 순 정렬 (크기가 비슷한 계수끼리 scale 공유)
    picked = spec.gather(-2, idx)
    parts = torch.stack([picked.real, picked.imag], dim=-2)  # (B, H, n, keep, 2, D)
    qfn = quantize_asym if coef_asym else quantize
    if coef_group is None or coef == "bf16" or coef_group >= keep:
        parts, coef_bytes = qfn(parts, coef, dim=(-3, -2))  # scale: (블록, 채널)마다 하나
    else:
        pad = -keep % coef_group
        grouped = torch.nn.functional.pad(parts, (0, 0, 0, 0, 0, pad))
        grouped = grouped.reshape(B, H, n_blk, -1, coef_group, 2, D)
        grouped, coef_bytes = qfn(grouped, coef, dim=(-3, -2))
        coef_bytes -= B * H * n_blk * D * 2 * pad * _BITS[coef] // 8  # 패딩은 저장 대상 아님
        parts = grouped.reshape(B, H, n_blk, keep + pad, 2, D)[..., :keep, :, :]
    nbytes += coef_bytes
    spec_hat = torch.zeros_like(spec).scatter(-2, idx, torch.complex(parts[..., 0, :], parts[..., 1, :]))
    out = torch.fft.irfft(spec_hat, n=block, dim=-2).reshape(B, H, n_blk * block, D)
    return torch.cat([out.to(x.dtype), tail], dim=-2), nbytes


def quantize_asym(v: torch.Tensor, coef: str, dim) -> tuple[torch.Tensor, int]:
    """비대칭 min-max 양자화 (2^bits 단계 전부 사용). 그룹마다 min/max를 bf16으로 저장"""
    if coef == "bf16":
        return v.to(torch.bfloat16).float(), v.numel() * 2
    levels = 2 ** _BITS[coef] - 1
    lo = v.amin(dim=dim, keepdim=True).to(torch.bfloat16).float()
    hi = v.amax(dim=dim, keepdim=True).to(torch.bfloat16).float()
    step = ((hi - lo) / levels).clamp_min(1e-8)
    q = ((v - lo) / step).round().clamp(0, levels)
    return q * step + lo, v.numel() * _BITS[coef] // 8 + lo.numel() * 4


def raw_codec(x: torch.Tensor, which: str, coef: str = "int8", group: int = 64) -> tuple[torch.Tensor, int]:
    """기준선: FFT 없이 KV를 그대로 양자화 (KIVI 방식, 비대칭 min-max)

    K는 채널마다 토큰 `group`개, V는 토큰마다 채널 `group`개 단위로 min/max 저장
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
    """비교용: head_dim 축 energy top-k (2~8절 방식). 토큰·head마다 bitmask + 계수"""
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
    """sequence 축 계수 선택 학습. score = log|X| + bias[layer, k|v][head, freq, channel]

    - bias 0 초기화 → energy top-k와 같은 지점에서 학습 시작, energy 대비 개선폭 = 학습 효과
    - 크기 정보(|X|)는 유지하고 downstream에 중요한 주파수·채널만 bias로 보정
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
    """학습용. forward는 learned top-k hard mask, backward는 sigmoid soft mask의 기울기 (STE)"""
    B, H, S, D = x.shape
    n_blk = S // block
    n_freq = block // 2 + 1
    keep = max(1, min(n_freq, round(keep_ratio * n_freq)))
    body = x[..., : n_blk * block, :].float().reshape(B, H, n_blk, block, D)
    spec = torch.fft.rfft(body, dim=-2)
    scores = spec.abs().clamp_min(1e-6).log() + bias[:, None]
    top = scores.topk(min(keep + 1, n_freq), dim=-2)
    hard = torch.zeros_like(scores).scatter(-2, top.indices[..., :keep, :], 1.0)
    if keep < n_freq:  # 경계 = keep번째와 keep+1번째 점수의 중간
        thr = top.values[..., keep - 1 : keep + 1, :].mean(-2, keepdim=True).detach()
    else:
        thr = top.values[..., -1:, :].detach() - 1.0
    soft = torch.sigmoid((scores - thr) / tau)
    mask = hard + soft - soft.detach()
    out = torch.fft.irfft(spec * mask, n=block, dim=-2).reshape(B, H, n_blk * block, D)
    return torch.cat([out.to(x.dtype), x[..., n_blk * block :, :]], dim=-2)


class FreqSelector(nn.Module):
    """토큰 하나의 head 벡터로 주파수별 중요도를 계산하는 MLP

    - 마지막 Linear를 weight 0, bias `init_logit`으로 초기화
    - 학습 시작 시 mask는 모든 주파수에서 sigmoid(init_logit) ~= 1 → student 출력이 teacher와 거의 동일
    - 이후 sparsity loss로 mask를 점차 낮추는 방식이라 초기화 직후 성능 저하 없음
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
        """mask 이전의 logit. Top-K 선택 기준값"""
        return self.net(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scores(x) / self.temperature)


class KVFreqCompressor(nn.Module):
    """레이어별 Selector_K / Selector_V 컨테이너

    - K와 V는 역할이 달라 selector 분리 (`share_kv=True`로 공유 가능)
    - `mode`만 바꿔 같은 selector로 Stage 1(soft)과 Stage 2(hard top-k, 정적 low-pass,
      random) 실험 모두 수행 가능

    mode:
        "off"       압축 없음 (= teacher)
        "soft"      sigmoid mask. 학습 단계에서 사용
        "topk"      selector 점수 상위 keep_ratio 만 1, 나머지 0 (토큰·head별 적응)
        "threshold" sigmoid mask > threshold 인 주파수만 1
        "lowpass"   낮은 주파수 keep_ratio 만 1 (학습 안 하는 정적 기준선)
        "energy"    |계수|가 큰 순으로 keep_ratio 만 1 (학습 없는 적응형 상한선)
        "random"    무작위 keep_ratio 만 1 (대조군)

    - `targets`로 K나 V만, `active_layers`로 일부 레이어만 압축 가능. 민감한 위치 탐색용
    - `renorm=True`: 재구성 벡터의 norm을 원본 norm으로 복원. 주파수를 버리면 에너지도 함께
      줄어드는데, 크기 변화만으로 출력이 달라지는지 따로 확인하는 옵션
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

        # 런타임 설정. 캐시 안에서 읽는 값이라 파라미터 대신 일반 속성
        self.mode = "soft"
        self.keep_ratio = 1.0
        self.threshold = 0.5
        self.straight_through = False  # hard mask에 soft gradient 전달 여부
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
        """상위 keep개 주파수만 1인 0/1 mask"""
        hard = torch.zeros_like(scores)
        idx = scores.topk(keep, dim=-1).indices
        return hard.scatter(-1, idx, 1.0)

    def make_mask(
        self, x: torch.Tensor, which: str, layer_idx: int, spec: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (..., D) -> mask: (..., F). `spec`는 energy 모드에서만 쓰는 rFFT 결과"""
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
            hard = hard + soft - soft.detach()  # forward는 hard, backward는 soft
        return hard

    # --------------------------------------------------------------- compress
    def compress(
        self, x: torch.Tensor, which: str, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, H, S, D) -> 같은 모양의 재구성 + mask (B, H, S, F)"""
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
                    "mask": mask,  # loss 계산에 필요해 graph 유지
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
        """sparsity loss에 쓰는 전체 평균 사용률. graph 연결 상태"""
        if not self._log:
            raise RuntimeError("log 없음. forward 먼저 실행 필요")
        return torch.stack([e["mask"].mean() for e in self._log]).mean()

    @torch.no_grad()
    def usage_table(self) -> dict[tuple[int, str], float]:
        """(layer, 'k'|'v') -> 평균 mask 값"""
        return {(e["layer"], e["which"]): float(e["mask"].mean()) for e in self._log}

    @torch.no_grad()
    def per_token_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """해당 레이어의 토큰별 사용률 (B, H, S) -> (S,) 평균"""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=-1).mean(dim=(0, 1)).float().cpu()
        raise KeyError((layer_idx, which))

    @torch.no_grad()
    def per_freq_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """해당 레이어의 주파수별 평균 사용률 (F,)"""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=(0, 1, 2)).float().cpu()
        raise KeyError((layer_idx, which))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class FFTCompressedCache(DynamicCache):
    """DynamicCache와 동일하게 동작하되, 저장 직전에 KV를 spectral filter에 통과

    Stage 1/2용. 저장 텐서 모양이 원본과 같아(재구성된 D차원 실수 벡터) 정보량만 감소,
    바이트는 동일. 바이트 절감은 `SparseSpectralCache`에서 처리
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
    """Stage 3: 전체 spectrum 대신 (주파수 index, complex 계수)만 저장하는 캐시

    토큰·head마다 계수 `keep`개를 골라 아래 형식으로 저장

        idx  : int16      (B, H, S, keep)
        coef : 실수부/허수부 각각 `coef_dtype` (B, H, S, keep)

    - attention에 넘길 때마다 0으로 채운 spectrum에 scatter 후 irFFT로 복원
    - 압축 상태로 저장, 읽을 때 복원하는 방식의 동작 확인용
    - `footprint_bytes()`: 현재 저장 중인 실제 바이트 반환
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

    # ------------------------------------------------------------ 내부 헬퍼
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

    # ------------------------------------------------------------------ 캐시
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        dtype = key_states.dtype
        bucket = self.store.setdefault(layer_idx, {})
        for which, states in (("k", key_states), ("v", value_states)):
            packed = self._encode(states, which, layer_idx)
            if which in bucket:
                self._append(bucket[which], packed)
            else:
                bucket[which] = packed

        # 길이·마스크 계산은 부모(DynamicCache)에 위임. 부모에는 head_dim 1짜리 더미만
        # 전달하고, 실제 KV는 self.store의 희소 표현으로만 보관
        stub = key_states.new_zeros((*key_states.shape[:-1], 1))
        super().update(stub, stub, layer_idx, *args, **kwargs)

        keys = self._decode(bucket["k"], dtype)
        values = self._decode(bucket["v"], dtype)
        return keys, values

    def footprint_bytes(self) -> int:
        """현재 저장 중인 실제 바이트 수"""
        total = 0
        for bucket in self.store.values():
            for packed in bucket.values():
                for tensor in packed.values():
                    total += tensor.numel() * tensor.element_size()
        return total
