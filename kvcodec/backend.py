"""vLLM CUSTOM attention backend: our 4/8-bit KV cache formats (real memory saving)

- `customize_spec` declares the cache cell size -> vLLM allocates the cache with it
    k8v4: 198 bytes (0.387x of the 512-byte bf16 cell)
    k4v4: 136 bytes (0.266x), needs a channel-mean table
- write (`do_kv_cache_update`): `kernel.write` packs the K/V of new tokens into cache pages (Triton)
- read (`forward`): `kernel.attention` computes attention on the packed cells without unpacking (Triton)

Environment variables
- KVCODEC       k8v4 (default) | k4v4
- KVCODEC_READ  triton (default) | torch (reference: unpack the whole context with `codec.unpack` every step)
- KVCODEC_WRITE triton (default) | torch (reference: `codec.pack`)
- KVCODEC_DOT   tf32 (default) | ieee (dot precision of the kernel, see `kernel.py`)
- KVCODEC_MEAN  path of the channel-mean table made by `kvcodec.calibrate` (used by k4v4)
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from kvcodec import codec, kernel

_READ = os.environ.get("KVCODEC_READ", "triton")
_WRITE = os.environ.get("KVCODEC_WRITE", "triton")
if not {_READ, _WRITE} <= {"triton", "torch"}:
    raise ValueError("KVCODEC_READ / KVCODEC_WRITE: triton | torch")


def _mode() -> str:
    mode = os.environ.get("KVCODEC", "k8v4")
    if mode not in codec.MODES:
        raise ValueError(f"KVCODEC={mode!r}: expected one of {sorted(codec.MODES)}")
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
        self._mean_v_heads: tuple[torch.Tensor | None] | None = None
        self._means_rot: tuple[torch.Tensor | None, ...] | None = None

    def _layer_means(self, layer, device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Channel means (H, D), only for the sides stored in 4 bits"""
        if self._means is None:
            table = _mean_table()
            found = re.findall(r"\d+", getattr(layer, "layer_name", "") or "")
            idx = int(found[0]) if found else 0
            pick = lambda which, bits: (table[which][idx].to(device=device, dtype=torch.float32)
                                        if table is not None and bits == 4 else None)
            self._means = (pick("k", self._k_bits), pick("v", self._v_bits))
        return self._means

    # ------------------------------------------------------------------ write
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """key/value: (num_tokens, H, D). kv_cache: (num_blocks, H, block_size, cell) viewed as bytes"""
        if _WRITE == "torch":
            return self._write_torch(layer, key, value, kv_cache, slot_mapping)
        n = slot_mapping.shape[0]
        if self._means_rot is None:
            self._means_rot = tuple(None if m is None else kernel.rotate_mean(m)
                                    for m in self._layer_means(layer, key.device))
        kernel.write(
            key[:n].view(n, self.num_kv_heads, self.head_size), value[:n].view(n, self.num_kv_heads, self.head_size),
            kv_cache.view(torch.uint8), slot_mapping, k_bits=self._k_bits, v_bits=self._v_bits,
            mean_k_rot=self._means_rot[0], mean_v_rot=self._means_rot[1],
        )

    def _write_torch(self, layer, key, value, kv_cache, slot_mapping):
        """Reference: pack with `codec.pack` and store"""
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
        cache[slots // block_size, :, slots % block_size] = cell  # (n, H, cell)

    # ------------------------------------------------------------------ read
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        if _READ == "torch":
            return self._forward_torch(layer, query, kv_cache, attn_metadata, output)
        n = attn_metadata.num_actual_tokens
        if self._mean_v_heads is None:  # only the V mean is needed (the K mean cancels in the softmax)
            mean_v = self._layer_means(layer, query.device)[1]
            n_rep = self.num_heads // self.num_kv_heads
            self._mean_v_heads = (mean_v.repeat_interleave(n_rep, dim=0) if mean_v is not None else None,)
        # segments are for decoding only: they pay off with one query per sequence and a long context
        decode = attn_metadata.max_query_len == 1 and attn_metadata.max_seq_len > 512 and attn_metadata.seq_lens.shape[0] <= 16
        kernel.attention(
            query[:n].view(n, self.num_heads, self.head_size), kv_cache.view(torch.uint8),
            output[:n].view(n, self.num_heads, self.head_size),
            query_start_loc=attn_metadata.query_start_loc, seq_lens=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table, scale=self.scale, k_bits=self._k_bits, v_bits=self._v_bits,
            mean_v=self._mean_v_heads[0], num_segments=kernel.NUM_SEGMENTS if decode else 1,
        )
        return output

    def _forward_torch(self, layer, query, kv_cache, attn_metadata, output):
        """Reference: per sequence, unpack the whole context with `codec.unpack` and run standard attention"""
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
            pages = cache[attn_metadata.block_table[i, :n_blk]]  # (n_blk, H, block_size, cell)
            pages = pages.transpose(0, 1).reshape(self.num_kv_heads, n_blk * block_size, -1)[:, :seq_len]
            k = codec.unpack(pages[..., : self._k_bytes], self._k_bits, mean_k, query.dtype)  # (H_kv, S, D)
            v = codec.unpack(pages[..., self._k_bytes :], self._v_bits, mean_v, query.dtype)
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=0)
                v = v.repeat_interleave(n_rep, dim=0)
            q = q_all[qs:qe].transpose(0, 1)  # (H, q_len, D)
            mask = None
            if q_len > 1:  # position of query j = seq_len - q_len + j; it attends to keys at or before it
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
