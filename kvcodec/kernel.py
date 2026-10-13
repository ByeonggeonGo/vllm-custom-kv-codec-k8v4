"""Triton kernels for the packed KV cache: attention that reads cells without unpacking (`attention`)

and a packing writer (`write`). No vLLM dependency. The read kernel is described here; see `_pack_side`
for the writer.

Idea: move the rotation and the mean removal to the query and the output instead of undoing them per key
    stored   x_rot = ((x - mean) * sign) @ H  ->  codes c, x_rot ~ c * step + lo  (8-bit: c * scale)
    score    q . k = q_rot . k_rot + q . mean_k,  q_rot = (q * sign) @ H
             q . mean_k is the same constant for every key of a query -> cancels in the softmax, so the
             K mean is not needed to read
             q_rot . k_rot = step * (q_rot . c) + lo * sum(q_rot)   -> one dot product with integer codes
    output   sum_t p_t v_t = (sum_t p_t v_rot_t) @ H * sign + mean_v   (sum p = 1)
             -> accumulate in the rotated space, un-rotate once per query
    4-bit codes are read as [D/2 low nibbles | D/2 high nibbles]. The resulting channel order is folded
    into row / column permutations of the rotation matrices (`_matrices`) -> no nibble interleave in the kernel

Grid
    (query block, KV head, segment). One program handles BLOCK_Q queries x the query heads sharing a KV head
    Segments: decoding (one query) launches few programs and leaves the GPU idle, so the context is split
    into NUM_SEGMENTS ranges computed in parallel; `_merge_kernel` combines their (sum, max, denominator)

Dot precision: TF32 tensor cores by default (same as vLLM's kernels; query mantissa rounded to 10 bits)
    KVCODEC_DOT=ieee selects exact float32 dot products (prefill about 2.7x slower)

Scope: causal mask only (no sliding window, ALiBi, sinks or softcap)
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kvcodec import codec

TILE = 32  # keys read per iteration
NUM_SEGMENTS = 16
EXACT_DOT = os.environ.get("KVCODEC_DOT", "tf32") == "ieee"
_MAT: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _dot(a, b, EXACT: tl.constexpr):
    if EXACT:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b)


@triton.jit
def _load_codes(ptr, cell, offs_d, mask, BITS: tl.constexpr, HALF: tl.constexpr):
    """Quantization codes as float32, shaped by the broadcast of cell and offs_d"""
    if BITS == 4:
        byte = tl.load(ptr + cell + offs_d % HALF, mask=mask, other=0).to(tl.int32)
        return ((byte >> (4 * (offs_d // HALF))) & 0xF).to(tl.float32)
    else:
        return tl.load(ptr + cell + offs_d, mask=mask, other=0).to(tl.int8, bitcast=True).to(tl.float32)


@triton.jit
def _load_bf16(ptr, offs, mask):
    """Two bytes -> bf16 -> float32 (bf16 bits = upper 16 bits of float32)"""
    b0 = tl.load(ptr + offs, mask=mask, other=0).to(tl.int32)
    b1 = tl.load(ptr + offs + 1, mask=mask, other=0).to(tl.int32)
    return ((b0 | (b1 << 8)) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _attn_kernel(
    out_ptr,  # NUM_SEGMENTS == 1: (N, H, D) final output. Otherwise (N, H, S, D) per-segment sums
    part_m_ptr,  # (N, H, S) per-segment score maximum
    part_l_ptr,  # (N, H, S) per-segment denominator
    q_ptr,  # (N, H, D) float32, rotated query
    cache_ptr,  # (blocks, H_kv, block_size, cell) uint8
    block_table_ptr,
    seq_lens_ptr,
    query_start_ptr,
    scale,
    num_seqs,
    block_table_stride,
    stride_blk,
    stride_head,
    stride_slot,
    NUM_HEADS: tl.constexpr,
    N_REP: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    K_BYTES: tl.constexpr,
    NUM_SEG: tl.constexpr,
    EXACT: tl.constexpr,
):
    q_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    seg = tl.program_id(2)
    HALF: tl.constexpr = HEAD_SIZE // 2

    # query block -> sequence. The blocks of sequence i start at query_start[i] // BLOCK_Q + i
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        if tl.load(query_start_ptr + mid) // BLOCK_Q + mid <= q_block:
            left = mid + 1
        else:
            right = mid
    seq = left - 1
    q_start = tl.load(query_start_ptr + seq)
    q_len = tl.load(query_start_ptr + seq + 1) - q_start
    seq_len = tl.load(seq_lens_ptr + seq)
    q_block_local = q_block - (q_start // BLOCK_Q + seq)
    if q_block_local * BLOCK_Q >= q_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local * BLOCK_Q + offs_m // N_REP
    head = kv_head * N_REP + offs_m % N_REP
    row_mask = (query_pos < q_len) & (offs_m < BLOCK_Q * N_REP)
    row = ((q_start + query_pos) * NUM_HEADS + head).to(tl.int64)

    Q = tl.load(q_ptr + row[:, None] * HEAD_SIZE + offs_d[None, :], mask=row_mask[:, None], other=0.0)
    q_sum = tl.sum(Q, axis=1)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    # absolute position of query j = context_len + j; it attends to keys at or before it
    context_len = seq_len - q_len
    max_prefix = tl.minimum(context_len + q_block_local * BLOCK_Q + BLOCK_Q, seq_len)
    num_tiles = (max_prefix + TILE_SIZE - 1) // TILE_SIZE
    tiles_per_seg = (num_tiles + NUM_SEG - 1) // NUM_SEG
    tile_lo = seg * tiles_per_seg
    tile_hi = tl.minimum(tile_lo + tiles_per_seg, num_tiles)

    for j in range(tile_lo, tile_hi):
        pos = j * TILE_SIZE + offs_t
        t_mask = pos < max_prefix
        blk = tl.load(block_table_ptr + seq * block_table_stride + pos // BLOCK_SIZE, mask=t_mask, other=0)
        cell = blk.to(tl.int64) * stride_blk + kv_head * stride_head + (pos % BLOCK_SIZE) * stride_slot

        # scores: dot of the rotated query with the K codes, scales applied after the dot
        K = _load_codes(cache_ptr, cell[None, :], offs_d[:, None], t_mask[None, :], K_BITS, HALF)  # (D, T)
        S = _dot(Q, K, EXACT)
        if K_BITS == 4:
            k_lo = _load_bf16(cache_ptr, cell + HALF, t_mask)
            k_step = _load_bf16(cache_ptr, cell + HALF + 2, t_mask)
            S = (S * k_step[None, :] + q_sum[:, None] * k_lo[None, :]) * scale
        else:
            k_scale = _load_bf16(cache_ptr, cell + HEAD_SIZE, t_mask)
            S = S * (k_scale[None, :] * scale)
        visible = row_mask[:, None] & (pos[None, :] <= context_len + query_pos[:, None])
        S = tl.where(visible, S, float("-inf"))

        # online softmax. Rows with a fully masked tile keep M = -inf (0 is used in the exponents only)
        m_new = tl.maximum(M, tl.max(S, axis=1))
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        P = tl.exp(S - m_safe[:, None])
        alpha = tl.exp(M - m_safe)
        L = L * alpha + tl.sum(P, axis=1)
        M = m_new

        # values: accumulate the V codes in the rotated space
        v_cell = cell + K_BYTES
        V = _load_codes(cache_ptr, v_cell[:, None], offs_d[None, :], t_mask[:, None], V_BITS, HALF)  # (T, D)
        if V_BITS == 4:
            v_lo = _load_bf16(cache_ptr, v_cell + HALF, t_mask)
            v_step = _load_bf16(cache_ptr, v_cell + HALF + 2, t_mask)
            acc = acc * alpha[:, None] + _dot(P * v_step[None, :], V, EXACT) + tl.sum(P * v_lo[None, :], axis=1)[:, None]
        else:
            v_scale = _load_bf16(cache_ptr, v_cell + HEAD_SIZE, t_mask)
            acc = acc * alpha[:, None] + _dot(P * v_scale[None, :], V, EXACT)

    if NUM_SEG == 1:
        tl.store(out_ptr + row[:, None] * HEAD_SIZE + offs_d[None, :], acc / L[:, None], mask=row_mask[:, None])
    else:
        part = row * NUM_SEG + seg
        tl.store(out_ptr + part[:, None] * HEAD_SIZE + offs_d[None, :], acc, mask=row_mask[:, None])
        tl.store(part_m_ptr + part, M, mask=row_mask)
        tl.store(part_l_ptr + part, L, mask=row_mask)


@triton.jit
def _merge_kernel(out_ptr, part_ptr, part_m_ptr, part_l_ptr, HEAD_SIZE: tl.constexpr, NUM_SEG: tl.constexpr):
    """Merge the per-segment online softmax states. Grid = (tokens x heads,)"""
    row = tl.program_id(0).to(tl.int64)
    offs_s = row * NUM_SEG + tl.arange(0, NUM_SEG)
    offs_d = tl.arange(0, HEAD_SIZE)
    m = tl.load(part_m_ptr + offs_s)
    w = tl.exp(m - tl.max(m, axis=0))  # empty segment: m = -inf -> weight 0
    acc = tl.load(part_ptr + offs_s[:, None] * HEAD_SIZE + offs_d[None, :])
    total = tl.sum(acc * w[:, None], axis=0) / tl.sum(tl.load(part_l_ptr + offs_s) * w, axis=0)
    tl.store(out_ptr + row * HEAD_SIZE + offs_d, total)


@triton.jit
def _bf16_bits(x):
    """float32 -> value rounded to bf16 (as float32) and its 16 bits (as int32)"""
    r = x.to(tl.bfloat16)
    return r.to(tl.float32), r.to(tl.int16, bitcast=True).to(tl.int32)


@triton.jit
def _store_bf16(ptr, offs, bits, mask):
    tl.store(ptr + offs, (bits & 0xFF).to(tl.uint8), mask=mask)
    tl.store(ptr + offs + 1, ((bits >> 8) & 0xFF).to(tl.uint8), mask=mask)


@triton.jit
def _pack_side(x_ptr, rot_ptr, mean_ptr, cache_ptr, row, head, cell, mask,
               BITS: tl.constexpr, USE_MEAN: tl.constexpr, HEAD_SIZE: tl.constexpr):
    """Rotate, quantize and store one side (K or V). Same format as `codec.pack`

    row: start offset of each input vector (BLOCK_N,)
    rot_ptr: (D, D) float32, +-1 Hadamard with the signs folded in (unnormalized, so exact under TF32)
    mean_ptr: (H_kv, D) float32, rotated channel mean
    """
    HALF: tl.constexpr = HEAD_SIZE // 2
    offs_d = tl.arange(0, HEAD_SIZE)
    X = tl.load(x_ptr + row[:, None] + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    norm = 1.0 / tl.sqrt(tl.full([1], HEAD_SIZE, dtype=tl.float32))
    if BITS == 4:
        # rotate even and odd channels separately -> low / high nibble of one byte
        offs_h = tl.arange(0, HALF)
        even = tl.dot(X, tl.load(rot_ptr + offs_d[:, None] * HEAD_SIZE + 2 * offs_h[None, :])) * norm
        odd = tl.dot(X, tl.load(rot_ptr + offs_d[:, None] * HEAD_SIZE + 2 * offs_h[None, :] + 1)) * norm
        if USE_MEAN:
            even -= tl.load(mean_ptr + head * HEAD_SIZE + 2 * offs_h)[None, :]
            odd -= tl.load(mean_ptr + head * HEAD_SIZE + 2 * offs_h + 1)[None, :]
        lo, lo_bits = _bf16_bits(tl.minimum(tl.min(even, axis=1), tl.min(odd, axis=1)))
        hi = tl.maximum(tl.max(even, axis=1), tl.max(odd, axis=1))
        step, step_bits = _bf16_bits(tl.maximum((hi - lo) / 15, 1e-6))
        c0 = tl.clamp(tl.floor((even - lo[:, None]) / step[:, None] + 0.5), 0.0, 15.0).to(tl.int32)
        c1 = tl.clamp(tl.floor((odd - lo[:, None]) / step[:, None] + 0.5), 0.0, 15.0).to(tl.int32)
        tl.store(cache_ptr + cell[:, None] + offs_h[None, :], (c0 | (c1 << 4)).to(tl.uint8), mask=mask[:, None])
        _store_bf16(cache_ptr, cell + HALF, lo_bits, mask)
        _store_bf16(cache_ptr, cell + HALF + 2, step_bits, mask)
    else:
        R = tl.dot(X, tl.load(rot_ptr + offs_d[:, None] * HEAD_SIZE + offs_d[None, :])) * norm
        if USE_MEAN:
            R -= tl.load(mean_ptr + head * HEAD_SIZE + offs_d)[None, :]
        scale, scale_bits = _bf16_bits(tl.maximum(tl.max(tl.abs(R), axis=1) / 127, 1e-6))
        c = tl.clamp(tl.floor(R / scale[:, None] + 0.5), -127.0, 127.0).to(tl.int8)
        tl.store(cache_ptr + cell[:, None] + offs_d[None, :], c.to(tl.uint8, bitcast=True), mask=mask[:, None])
        _store_bf16(cache_ptr, cell + HEAD_SIZE, scale_bits, mask)


@triton.jit
def _write_kernel(
    key_ptr,  # (N, H_kv, D), token and head axes addressed by stride (vLLM's value is a non-contiguous slice of qkv)
    value_ptr,
    cache_ptr,  # (blocks, H_kv, block_size, cell) uint8
    slot_ptr,  # (N,), negative entries are skipped
    rot_ptr,
    mean_k_ptr,
    mean_v_ptr,
    num_tokens,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    stride_blk,
    stride_head,
    stride_slot,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    K_BYTES: tl.constexpr,
    USE_MEAN_K: tl.constexpr,
    USE_MEAN_V: tl.constexpr,
):
    """Grid = (token block, KV head). One program packs and stores K and V of BLOCK_N tokens"""
    head = tl.program_id(1)
    tok = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    in_range = tok < num_tokens
    slot = tl.load(slot_ptr + tok, mask=in_range, other=-1).to(tl.int64)
    mask = in_range & (slot >= 0)
    k_row = tok.to(tl.int64) * stride_k_tok + head * stride_k_head
    v_row = tok.to(tl.int64) * stride_v_tok + head * stride_v_head
    cell = (slot // BLOCK_SIZE) * stride_blk + head * stride_head + (slot % BLOCK_SIZE) * stride_slot
    _pack_side(key_ptr, rot_ptr, mean_k_ptr, cache_ptr, k_row, head, cell, mask, K_BITS, USE_MEAN_K, HEAD_SIZE)
    _pack_side(value_ptr, rot_ptr, mean_v_ptr, cache_ptr, v_row, head, cell + K_BYTES, mask, V_BITS, USE_MEAN_V, HEAD_SIZE)


def rotate_mean(mean: torch.Tensor) -> torch.Tensor:
    """Rotate the channel mean (H_kv, D) for the write kernel: subtracting after the rotation is equivalent"""
    sign, had = codec.rotation(mean.shape[-1], mean.device)
    return ((mean.float() * sign) @ had).contiguous()


def write(key: torch.Tensor, value: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor, *,
          k_bits: int, v_bits: int, mean_k_rot: torch.Tensor | None = None,
          mean_v_rot: torch.Tensor | None = None) -> None:
    """Pack the K/V of new tokens into cache cells

    key, value: (N, H_kv, D), only the last axis must be contiguous. cache: (blocks, H_kv, block_size, cell)
    uint8. slot_mapping: (N,)
    mean_*_rot: channel means passed through `rotate_mean`, or None
    """
    n, num_kv_heads, head_size = key.shape
    assert key.stride(2) == 1 and value.stride(2) == 1 and cache.stride(3) == 1
    block_n = 16
    rot_key = (head_size, "write", str(key.device))
    if rot_key not in _MAT:
        sign, had = codec.rotation(head_size, key.device)
        _MAT[rot_key] = ((sign[:, None] * had).sign().contiguous(),) * 2
    rot = _MAT[rot_key][0]
    _write_kernel[(triton.cdiv(n, block_n), num_kv_heads)](
        key, value, cache, slot_mapping, rot,
        mean_k_rot if mean_k_rot is not None else rot, mean_v_rot if mean_v_rot is not None else rot,
        n, key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2),
        HEAD_SIZE=head_size, BLOCK_SIZE=cache.shape[2], BLOCK_N=block_n,
        K_BITS=k_bits, V_BITS=v_bits, K_BYTES=codec.side_bytes(head_size, k_bits),
        USE_MEAN_K=mean_k_rot is not None, USE_MEAN_V=mean_v_rot is not None,
    )


def _matrices(head_size: int, k_bits: int, v_bits: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(query rotation, output un-rotation) matrices, including the signs and the 4-bit nibble order
    (even channels | odd channels)"""
    key = (head_size, k_bits, v_bits, str(device))
    if key not in _MAT:
        sign, had = codec.rotation(head_size, device)
        order = lambda bits: (torch.cat([torch.arange(0, head_size, 2), torch.arange(1, head_size, 2)])
                              if bits == 4 else torch.arange(head_size)).to(device)
        q_mat = (sign[:, None] * had)[:, order(k_bits)]  # q @ q_mat = ((q * sign) @ had)[perm]
        o_mat = had[order(v_bits)] * sign[None, :]  # acc[perm] @ o_mat = (acc @ had) * sign
        _MAT[key] = (q_mat.contiguous(), o_mat.contiguous())
    return _MAT[key]


def attention(q: torch.Tensor, cache: torch.Tensor, out: torch.Tensor, *, query_start_loc: torch.Tensor,
              seq_lens: torch.Tensor, block_table: torch.Tensor, scale: float, k_bits: int, v_bits: int,
              mean_v: torch.Tensor | None = None, num_segments: int = 1, exact: bool | None = None) -> None:
    """Paged attention over the packed cache, written into `out`

    q, out: (N, H, D). cache: (blocks, H_kv, block_size, cell) uint8. mean_v: (H, D) or None
    query_start_loc (num_seqs + 1,), seq_lens (num_seqs,), block_table (num_seqs, num blocks): int32
    num_segments > 1 must be a power of two (for decoding)
    """
    n, num_heads, head_size = q.shape
    assert cache.stride(3) == 1
    num_kv_heads, block_size = cache.shape[1], cache.shape[2]
    n_rep = num_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(n_rep))
    block_q = block_m // n_rep
    num_seqs = query_start_loc.shape[0] - 1
    q_mat, o_mat = _matrices(head_size, k_bits, v_bits, q.device)
    q_rot = q.float() @ q_mat
    acc = torch.empty((n, num_heads, head_size), dtype=torch.float32, device=q.device)
    if num_segments > 1:
        part = torch.empty((n, num_heads, num_segments, head_size), dtype=torch.float32, device=q.device)
        part_m = torch.empty((n, num_heads, num_segments), dtype=torch.float32, device=q.device)
        part_l = torch.empty_like(part_m)
    else:
        part = part_m = part_l = acc  # unused placeholders

    _attn_kernel[(n // block_q + num_seqs, num_kv_heads, num_segments)](
        part, part_m, part_l, q_rot, cache, block_table, seq_lens, query_start_loc,
        scale, num_seqs, block_table.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
        NUM_HEADS=num_heads, N_REP=n_rep, HEAD_SIZE=head_size, BLOCK_SIZE=block_size,
        BLOCK_M=block_m, BLOCK_Q=block_q, TILE_SIZE=TILE, K_BITS=k_bits, V_BITS=v_bits,
        K_BYTES=codec.side_bytes(head_size, k_bits), NUM_SEG=num_segments,
        EXACT=EXACT_DOT if exact is None else exact,
    )
    if num_segments > 1:
        _merge_kernel[(n * num_heads,)](acc, part, part_m, part_l, HEAD_SIZE=head_size, NUM_SEG=num_segments)
    res = acc @ o_mat
    if mean_v is not None:
        res += mean_v
    out.copy_(res)
