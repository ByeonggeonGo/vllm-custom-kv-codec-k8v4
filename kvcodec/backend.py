"""vLLM CUSTOM attention 백엔드: 직접 구현한 4/8비트 KV 캐시 (실제 메모리 절감)

- `customize_spec`으로 캐시 칸 크기를 선언 → vLLM이 그 크기로 캐시 할당
    k8v4: 198바이트 (bf16 원본 512바이트의 0.387x)
    k4v4: 136바이트 (0.266x), 채널 평균 표 필요
- 쓰기 (`do_kv_cache_update`): 새 토큰의 K/V를 `codec.pack`으로 압축해 캐시 페이지에 기록
- 읽기 (`forward`): 시퀀스의 캐시 블록을 `codec.unpack`으로 풀어 표준 attention 계산

v1의 읽기는 매 스텝 문맥 전체를 풀어서 계산 (PyTorch). 전용 Triton 커널은 다음 단계

환경변수
- KVCODEC       k8v4 (기본) | k4v4
- KVCODEC_MEAN  `kvcodec.calibrate`가 만든 채널 평균 표 경로 (k4v4에서 사용)
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from kvcodec import codec


def _mode() -> str:
    mode = os.environ.get("KVCODEC", "k8v4")
    if mode not in codec.MODES:
        raise ValueError(f"KVCODEC={mode!r}: {sorted(codec.MODES)} 중 하나")
    return mode


_MEAN_TABLE: dict[str, torch.Tensor] | None = None


def _mean_table() -> dict[str, torch.Tensor] | None:
    global _MEAN_TABLE
    path = os.environ.get("KVCODEC_MEAN")
    if _MEAN_TABLE is None and path:
        _MEAN_TABLE = torch.load(path, map_location="cpu")
    return _MEAN_TABLE


class CodecAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k_bits, self._v_bits = codec.MODES[_mode()]
        self._k_bytes = codec.side_bytes(self.head_size, self._k_bits)
        self._means: tuple[torch.Tensor | None, torch.Tensor | None] | None = None

    def _layer_means(self, layer, device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """4비트로 저장하는 쪽에만 채널 평균 (H, D) 적용"""
        if self._means is None:
            table = _mean_table()
            found = re.findall(r"\d+", getattr(layer, "layer_name", "") or "")
            idx = int(found[0]) if found else 0
            pick = lambda which, bits: (table[which][idx].to(device=device, dtype=torch.float32)
                                        if table is not None and bits == 4 else None)
            self._means = (pick("k", self._k_bits), pick("v", self._v_bits))
        return self._means

    # ------------------------------------------------------------------ 쓰기
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """key/value: (num_tokens, H, D). kv_cache: (num_blocks, H, block_size, 칸) 바이트 뷰"""
        cache = kv_cache.view(torch.uint8)
        block_size = cache.shape[2]
        n = slot_mapping.shape[0]
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        if slots.numel() == 0:
            return
        mean_k, mean_v = self._layer_means(layer, key.device)
        k = key[:n].view(n, self.num_kv_heads, self.head_size)[valid]
        v = value[:n].view(n, self.num_kv_heads, self.head_size)[valid]
        cell = torch.cat([codec.pack(k, self._k_bits, mean_k), codec.pack(v, self._v_bits, mean_v)], dim=-1)
        cache[slots // block_size, :, slots % block_size] = cell  # (n, H, 칸)

    # ------------------------------------------------------------------ 읽기
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:  # 프로파일링 실행
            return output.fill_(0)
        cache = kv_cache.view(torch.uint8)
        block_size = cache.shape[2]
        n_rep = self.num_heads // self.num_kv_heads
        num_tokens = attn_metadata.num_actual_tokens
        mean_k, mean_v = self._layer_means(layer, query.device)
        if mean_k is not None:
            mean_k = mean_k[:, None]  # (H, 1, D)
        if mean_v is not None:
            mean_v = mean_v[:, None]
        q_all = query[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        out_all = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        starts = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()

        for i in range(len(starts) - 1):
            qs, qe = starts[i], starts[i + 1]
            q_len, seq_len = qe - qs, seq_lens[i]
            if q_len <= 0:
                continue
            n_blk = -(-seq_len // block_size)
            pages = cache[attn_metadata.block_table[i, :n_blk]]  # (n_blk, H, block_size, 칸)
            pages = pages.transpose(0, 1).reshape(self.num_kv_heads, n_blk * block_size, -1)[:, :seq_len]
            k = codec.unpack(pages[..., : self._k_bytes], self._k_bits, mean_k, query.dtype)  # (H_kv, S, D)
            v = codec.unpack(pages[..., self._k_bytes :], self._v_bits, mean_v, query.dtype)
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=0)
                v = v.repeat_interleave(n_rep, dim=0)
            q = q_all[qs:qe].transpose(0, 1)  # (H, q_len, D)
            mask = None
            if q_len > 1:  # 쿼리 j의 위치 = seq_len - q_len + j, 그 이하의 키만 참조
                pos = torch.arange(seq_len, device=q.device)
                last = torch.arange(seq_len - q_len, seq_len, device=q.device)
                mask = pos[None, :] <= last[:, None]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
            out_all[qs:qe] = o.transpose(0, 1)
        return output


class CodecAttentionBackend(TritonAttentionBackend):
    @classmethod
    def customize_spec(cls, spec):
        if spec.state_content_bytes is not None:
            return spec
        return replace(spec, state_content_bytes=codec.cell_bytes(spec.head_size, _mode()))

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[CodecAttentionImpl]:
        return CodecAttentionImpl
