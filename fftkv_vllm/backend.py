"""vLLM Triton attention 백엔드에 sequence 축 FFT 코덱을 끼운 CUSTOM 백엔드

동작
- vLLM 블록 크기를 FFT 블록 길이와 같게 실행 (`--block-size 256`) → vLLM 블록 하나 = FFT 블록 하나
- 블록이 가득 차고 그 토큰들이 전부 과거(context)가 된 시점에 해당 블록의 K/V를 `seq_codec`으로
  저장 포맷으로 변환 후 복원, 캐시에 다시 기록
- 덜 찬 마지막 블록과 이번 스텝의 새 토큰은 원본 유지
- attention 계산은 부모(Triton 커널) 그대로

한계 (v1)
- 캐시 텐서는 여전히 bf16. 저장 포맷의 바이트는 `seq_codec`이 센 값으로 집계만 함 (stats 파일)
- 품질(출력)은 압축 포맷 그대로 저장했을 때와 동일, 실제 GPU 메모리 절감은 미구현

환경변수
- FFTKV_MODE      learned | energy | lowpass | off   (기본 energy, selector 지정 시 learned)
- FFTKV_KEEP      남길 계수 비율 (기본 0.25)
- FFTKV_COEF      bf16 | int8 | int4 | int2          (기본 int4)
- FFTKV_COEF_GROUP scale 공유 계수 개수              (기본 8)
- FFTKV_SELECTOR  SeqFreqSelector state_dict 경로 (.pt)
- FFTKV_SKIP_BLOCKS 시퀀스 앞쪽에서 압축하지 않을 블록 수 (기본 0)
- FFTKV_STATS     통계 json 경로
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from custom_layers import seq_codec


@dataclass
class CodecConfig:
    mode: str = "energy"
    keep: float = 0.25
    coef: str = "int4"
    coef_group: int = 8
    selector: str | None = None
    skip_blocks: int = 0
    stats: str | None = None

    @classmethod
    def from_env(cls) -> "CodecConfig":
        selector = os.environ.get("FFTKV_SELECTOR") or None
        return cls(
            mode=os.environ.get("FFTKV_MODE", "learned" if selector else "energy"),
            keep=float(os.environ.get("FFTKV_KEEP", 0.25)),
            coef=os.environ.get("FFTKV_COEF", "int4"),
            coef_group=int(os.environ.get("FFTKV_COEF_GROUP", 8)),
            selector=selector,
            skip_blocks=int(os.environ.get("FFTKV_SKIP_BLOCKS", 0)),
            stats=os.environ.get("FFTKV_STATS") or None,
        )


# 프로세스 전체 집계 (레이어 인스턴스들이 공유)
_STATS = {"blocks": 0, "raw_bytes": 0, "stored_bytes": 0}
_SELECTOR_STATE: dict[str, torch.Tensor] | None = None


def _selector_bias(cfg: CodecConfig, layer_idx: int, which: str, device) -> torch.Tensor | None:
    global _SELECTOR_STATE
    if cfg.mode != "learned":
        return None
    if _SELECTOR_STATE is None:
        _SELECTOR_STATE = torch.load(cfg.selector, map_location="cpu")
    return _SELECTOR_STATE[f"bias.{layer_idx}_{which}"].to(device=device, dtype=torch.float32)


class FFTKVAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cfg = CodecConfig.from_env()
        self._done: torch.Tensor | None = None  # (num_blocks,) 이미 압축한 블록 표시
        self._layer_idx: int | None = None
        self._bias: dict[str, torch.Tensor | None] = {}

    # ------------------------------------------------------------------ 쓰기
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # 블록의 0번 슬롯에 새 토큰이 쓰이면 그 블록은 새로 할당된 것 → 압축 표시 해제
        if self._done is not None:
            block_size = kv_cache.shape[2]
            fresh = slot_mapping[(slot_mapping >= 0) & (slot_mapping % block_size == 0)] // block_size
            if fresh.numel():
                self._done[fresh] = False
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    # ------------------------------------------------------------------ 읽기
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if attn_metadata is not None and self._cfg.mode != "off" and kv_cache.numel() > 0:
            self._compress_full_blocks(layer, kv_cache, attn_metadata)
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)

    @torch.no_grad()
    def _compress_full_blocks(self, layer, kv_cache: torch.Tensor, md) -> None:
        """kv_cache: (num_blocks, H, block_size, 2 * head_size). 과거가 된 가득 찬 블록만 처리"""
        cfg = self._cfg
        block_size = kv_cache.shape[2]
        if self._done is None or self._done.shape[0] != kv_cache.shape[0]:
            self._done = torch.zeros(kv_cache.shape[0], dtype=torch.bool, device=kv_cache.device)
        query_lens = md.query_start_loc[1:] - md.query_start_loc[:-1]
        n_full = ((md.seq_lens[: query_lens.shape[0]] - query_lens) // block_size).tolist()
        if not any(n > cfg.skip_blocks for n in n_full):
            return

        ids = torch.cat([md.block_table[i, cfg.skip_blocks : n] for i, n in enumerate(n_full)
                         if n > cfg.skip_blocks])
        ids = torch.unique(ids[~self._done[ids]])
        if ids.numel() == 0:
            return

        if self._layer_idx is None:
            found = re.findall(r"\d+", getattr(layer, "layer_name", "") or "")
            self._layer_idx = int(found[0]) if found else 0
            for which in ("k", "v"):
                self._bias[which] = _selector_bias(cfg, self._layer_idx, which, kv_cache.device)

        hs = self.head_size
        blocks = kv_cache[ids]  # (n, H, block_size, 2 * hs)
        out, stored = [], 0
        for which, x in (("k", blocks[..., :hs]), ("v", blocks[..., hs:])):
            y, nbytes = seq_codec(x, cfg.keep, block=block_size, mode=cfg.mode, coef=cfg.coef,
                                  bias=self._bias[which], coef_group=cfg.coef_group)
            out.append(y)
            stored += nbytes
        kv_cache[ids] = torch.cat(out, dim=-1)
        self._done[ids] = True

        _STATS["blocks"] += int(ids.numel())
        _STATS["raw_bytes"] += blocks.numel() * blocks.element_size()
        _STATS["stored_bytes"] += stored
        if cfg.stats:
            with open(cfg.stats, "w") as f:
                json.dump({**_STATS, "ratio": _STATS["stored_bytes"] / max(1, _STATS["raw_bytes"]),
                           "config": cfg.__dict__}, f)


class FFTKVAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FFTKVAttentionImpl]:
        return FFTKVAttentionImpl
