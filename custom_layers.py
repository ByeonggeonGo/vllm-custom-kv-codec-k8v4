"""FFT 기반 adaptive KV compression에 필요한 레이어들.

흐름은 계획서 그대로다::

    K, V -> rFFT(head_dim 축) -> selector가 만든 mask -> irFFT -> 기존 attention

끼워넣는 지점은 attention 내부가 아니라 **캐시의 update()** 다. transformers v5의
`Qwen3Attention.forward`는 RoPE를 적용한 뒤 딱 한 줄로

    key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

를 호출하고, 그 반환값으로 attention을 계산한다. 즉 `DynamicCache.update`를 감싸면
"캐시에 들어가는 KV"와 "attention이 보는 KV"를 동시에 바꿀 수 있다. 모델 코드는
한 줄도 수정하지 않는다.

주의: `update`는 새로 들어온 토큰만 받고, 과거 토큰은 이미 압축된 상태로 저장되어
있으므로 이중 압축이 일어나지 않는다.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

__all__ = [
    "n_freq_of",
    "spectral_filter",
    "FreqSelector",
    "KVFreqCompressor",
    "FFTCompressedCache",
    "SparseSpectralCache",
]


def n_freq_of(head_dim: int) -> int:
    """실수 입력 길이 D에 대한 rFFT 계수 개수 (bin 0..D/2)."""
    return head_dim // 2 + 1


def spectral_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x를 head_dim 축으로 rFFT -> mask 곱 -> irFFT.

    x:    (..., D)   실수. bf16이면 fft가 안 되므로 내부에서 float32로 올린다.
    mask: (..., F)   F = D//2+1. [0, 1] 범위를 기대하지만 강제하지 않는다.
    반환:  (..., D)   x와 같은 dtype.
    """
    head_dim = x.shape[-1]
    spec = torch.fft.rfft(x.float(), dim=-1)
    spec = spec * mask.float()  # complex * real -> 실수부/허수부에 동일 배율
    return torch.fft.irfft(spec, n=head_dim, dim=-1).to(x.dtype)


class FreqSelector(nn.Module):
    """토큰 하나의 head 벡터를 보고 각 주파수의 중요도를 내는 작은 MLP.

    마지막 Linear의 weight를 0, bias를 `init_logit`으로 초기화한다. 그래서 학습
    시작 시점의 mask는 모든 주파수에서 sigmoid(init_logit) ~= 1 이고, student는
    teacher와 거의 같은 출력을 낸다. 여기서부터 sparsity loss가 mask를 깎아
    내려가는 구조라, 초기화 때문에 성능이 무너지는 일이 없다.
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
        """mask 이전의 logit. Top-K 선택은 이 값을 기준으로 한다."""
        return self.net(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scores(x) / self.temperature)


class KVFreqCompressor(nn.Module):
    """레이어마다 Selector_K / Selector_V 를 들고 있는 컨테이너.

    K와 V는 역할이 다르므로 selector를 따로 둔다 (`share_kv=True`로 묶을 수도 있다).
    `mode`를 바꾸면 같은 학습된 selector로 Stage 1(soft) / Stage 2(hard top-k, 정적
    low-pass, random) 실험을 모두 돌릴 수 있다.

    mode:
        "off"       압축 없음 (= teacher)
        "soft"      sigmoid mask. 학습 단계에서 사용
        "topk"      selector 점수 상위 keep_ratio 만 1, 나머지 0 (토큰·head별 적응)
        "threshold" sigmoid mask > threshold 인 주파수만 1
        "lowpass"   낮은 주파수 keep_ratio 만 1 (학습 안 하는 정적 기준선)
        "energy"    |계수|가 큰 순으로 keep_ratio 만 1 (학습 없는 적응형 상한선)
        "random"    무작위 keep_ratio 만 1 (대조군)

    `targets`로 K만/V만 압축할 수 있고, `active_layers`로 일부 레이어만 건드릴 수
    있다. 둘 다 "어디가 민감한가"를 재는 데 쓴다.

    `renorm=True`면 재구성 벡터의 norm을 원본 norm으로 되돌린다. 주파수를 버리면
    에너지가 같이 줄어드는데, 그 크기 변화만으로도 출력이 흔들리는지 분리해 보기
    위한 스위치다.
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

        # 런타임 설정. 캐시 안에서 읽히므로 파라미터가 아니라 평범한 속성이다.
        self.mode = "soft"
        self.keep_ratio = 1.0
        self.threshold = 0.5
        self.straight_through = False  # hard mask에 soft gradient를 흘릴지
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
        """상위 keep개 주파수만 1인 0/1 mask."""
        hard = torch.zeros_like(scores)
        idx = scores.topk(keep, dim=-1).indices
        return hard.scatter(-1, idx, 1.0)

    def make_mask(
        self, x: torch.Tensor, which: str, layer_idx: int, spec: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (..., D) -> mask: (..., F). `spec`는 energy 모드에서만 쓰는 rFFT 결과."""
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
        """(B, H, S, D) -> 같은 모양의 재구성 + mask (B, H, S, F)."""
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
                    "mask": mask,  # loss 계산에 필요하므로 graph를 유지한다
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
        """sparsity loss에 쓰는 전체 평균 사용률. graph가 붙어 있다."""
        if not self._log:
            raise RuntimeError("log가 비어 있다. forward를 먼저 돌려라.")
        return torch.stack([e["mask"].mean() for e in self._log]).mean()

    @torch.no_grad()
    def usage_table(self) -> dict[tuple[int, str], float]:
        """(layer, 'k'|'v') -> 평균 mask 값."""
        return {(e["layer"], e["which"]): float(e["mask"].mean()) for e in self._log}

    @torch.no_grad()
    def per_token_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """해당 레이어의 토큰별 사용률 (B, H, S) -> (S,) 평균."""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=-1).mean(dim=(0, 1)).float().cpu()
        raise KeyError((layer_idx, which))

    @torch.no_grad()
    def per_freq_usage(self, layer_idx: int, which: str = "k") -> torch.Tensor:
        """해당 레이어의 주파수별 평균 사용률 (F,)."""
        for e in self._log:
            if e["layer"] == layer_idx and e["which"] == which:
                return e["mask"].mean(dim=(0, 1, 2)).float().cpu()
        raise KeyError((layer_idx, which))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class FFTCompressedCache(DynamicCache):
    """DynamicCache와 똑같이 동작하되, 저장 직전에 KV를 spectral filter에 통과시킨다.

    Stage 1/2 용. 저장되는 텐서 모양은 원본과 같으므로(= 재구성된 D차원 실수 벡터)
    여기서 줄어드는 건 "정보량"이지 바이트 수는 아니다. 실제 바이트를 줄이는 건
    `SparseSpectralCache`.
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
    """Stage 3: 전체 spectrum이 아니라 (주파수 index, complex 계수)만 들고 있는 캐시.

    토큰·head마다 고정 개수 `keep`개의 계수를 고른다. 저장은

        idx  : int16      (B, H, S, keep)
        coef : 실수부/허수부 각각 `coef_dtype` (B, H, S, keep)

    attention에는 매번 0으로 채운 spectrum에 scatter 후 irFFT 해 돌려준다. 즉
    "압축된 상태로 저장, 읽을 때 복원"이 실제로 성립하는지 확인하는 용도다.
    `footprint_bytes()`가 지금 들고 있는 실제 바이트를 돌려준다.
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

        # 길이·마스크 계산은 부모(DynamicCache)에게 맡긴다. 다만 부모가 들고 갈
        # 텐서는 head_dim 1짜리 더미라, 실제 KV는 self.store의 희소 표현에만 있다.
        stub = key_states.new_zeros((*key_states.shape[:-1], 1))
        super().update(stub, stub, layer_idx, *args, **kwargs)

        keys = self._decode(bucket["k"], dtype)
        values = self._decode(bucket["v"], dtype)
        return keys, values

    def footprint_bytes(self) -> int:
        """지금 실제로 들고 있는 바이트 수."""
        total = 0
        for bucket in self.store.values():
            for packed in bucket.values():
                for tensor in packed.values():
                    total += tensor.numel() * tensor.element_size()
        return total
