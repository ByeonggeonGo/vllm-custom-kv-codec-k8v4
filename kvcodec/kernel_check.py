"""Check the Triton kernels: read against `codec.unpack` + PyTorch attention, write against `codec.pack`.
Also times the kernels on their own.

    python -m kvcodec.kernel_check          (needs a GPU and triton, not vLLM)

- batches: several sequences mixing prefill and decode, block table in random order
- combinations: mode (k8v4, k4v4) x segments (1, 16) x head layout (GQA 2x, 1x)
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from kvcodec import codec, kernel

DEV = "cuda"
BLOCK = 16


def make_batch(mode: str, cases: list[tuple[int, int]], num_heads: int, num_kv_heads: int, head_size: int, seed=0):
    """cases: list of (seq_len, q_len) -> kernel inputs and the original K/V of each sequence"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    k_bits, v_bits = codec.MODES[mode]
    rand = lambda *shape: torch.randn(*shape, device=DEV, generator=g)
    mean_k = rand(num_kv_heads, head_size) * 3 if k_bits == 4 else None
    mean_v = rand(num_kv_heads, head_size) if v_bits == 4 else None
    blocks_per_seq = [-(-s // BLOCK) for s, _ in cases]
    num_blocks = sum(blocks_per_seq) + 3
    # non-contiguous view with vLLM's memory order (block, slot, head, cell). Unused cells hold garbage
    cache = torch.randint(0, 256, (num_blocks, BLOCK, num_kv_heads, codec.cell_bytes(head_size, mode)),
                          dtype=torch.uint8, device=DEV, generator=g).transpose(1, 2)
    perm = torch.randperm(num_blocks, device=DEV, generator=g)
    block_table = torch.zeros(len(cases), max(blocks_per_seq), dtype=torch.int32, device=DEV)
    kvs, used = [], 0
    for i, (seq_len, _) in enumerate(cases):
        ids = perm[used : used + blocks_per_seq[i]]
        used += blocks_per_seq[i]
        block_table[i, : len(ids)] = ids.int()
        k = rand(seq_len, num_kv_heads, head_size) * (1 + 20 * (torch.arange(head_size, device=DEV) % 37 == 0))
        v = rand(seq_len, num_kv_heads, head_size)
        if mean_k is not None:
            k = k + mean_k
        if mean_v is not None:
            v = v + mean_v
        cell = torch.cat([codec.pack(k, k_bits, mean_k), codec.pack(v, v_bits, mean_v)], dim=-1)
        pos = torch.arange(seq_len, device=DEV)
        cache[ids[pos // BLOCK], :, pos % BLOCK] = cell
        kvs.append((k, v))
    q_lens = [q for _, q in cases]
    starts = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    q = rand(sum(q_lens), num_heads, head_size)
    return dict(q=q, cache=cache, block_table=block_table, query_start_loc=starts,
                seq_lens=torch.tensor([s for s, _ in cases], dtype=torch.int32, device=DEV),
                mean_k=mean_k, mean_v=mean_v, k_bits=k_bits, v_bits=v_bits, kvs=kvs, cases=cases)


def reference(b: dict, scale: float, from_cache: bool) -> torch.Tensor:
    """from_cache: use the cache unpacked by `codec.unpack` (must match the kernel). Otherwise the K/V
    before quantization"""
    q = b["q"]
    n_rep = q.shape[1] // b["cache"].shape[1]
    k_bytes = codec.side_bytes(q.shape[2], b["k_bits"])
    out, qs = torch.empty_like(q), 0
    for i, (seq_len, q_len) in enumerate(b["cases"]):
        if from_cache:
            pos = torch.arange(seq_len, device=DEV)
            cell = b["cache"][b["block_table"][i, pos // BLOCK].long(), :, pos % BLOCK]  # (S, H_kv, cell)
            k = codec.unpack(cell[..., :k_bytes], b["k_bits"], b["mean_k"])
            v = codec.unpack(cell[..., k_bytes:], b["v_bits"], b["mean_v"])
        else:
            k, v = b["kvs"][i]
        k = k.transpose(0, 1).repeat_interleave(n_rep, dim=0)  # (H, S, D)
        v = v.transpose(0, 1).repeat_interleave(n_rep, dim=0)
        pos = torch.arange(seq_len, device=DEV)
        mask = pos[None, :] <= torch.arange(seq_len - q_len, seq_len, device=DEV)[:, None]
        o = F.scaled_dot_product_attention(q[qs : qs + q_len].transpose(0, 1), k, v, attn_mask=mask, scale=scale)
        out[qs : qs + q_len] = o.transpose(0, 1)
        qs += q_len
    return out


def run_kernel(b: dict, scale: float, num_segments: int, exact: bool = False) -> torch.Tensor:
    out = torch.full_like(b["q"], float("nan"))
    n_rep = b["q"].shape[1] // b["cache"].shape[1]
    mean_v = None if b["mean_v"] is None else b["mean_v"].repeat_interleave(n_rep, dim=0)
    kernel.attention(b["q"], b["cache"], out, query_start_loc=b["query_start_loc"], seq_lens=b["seq_lens"],
                     block_table=b["block_table"], scale=scale, k_bits=b["k_bits"], v_bits=b["v_bits"],
                     mean_v=mean_v, num_segments=num_segments, exact=exact)
    return out


def check_write() -> None:
    """Write kernel vs `codec.pack`: fraction of identical bytes, reconstruction error"""
    print("write: cells written by the kernel vs codec.pack (byte match, relative RMS error kernel | pack)")
    g = torch.Generator(device=DEV).manual_seed(1)
    for num_kv_heads in (8, 4):
        for mode in codec.MODES:
            k_bits, v_bits = codec.MODES[mode]
            k_bytes = codec.side_bytes(128, k_bits)
            for n in (1, 37, 700):
                spike = 1 + 20 * (torch.arange(128, device=DEV) % 37 == 0)
                mean_k = torch.randn(num_kv_heads, 128, device=DEV, generator=g) * 3 if k_bits == 4 else None
                mean_v = torch.randn(num_kv_heads, 128, device=DEV, generator=g) if v_bits == 4 else None
                key = (torch.randn(n, num_kv_heads, 128, device=DEV, generator=g) * spike).bfloat16()
                # as in vLLM: value is a slice of a wider tensor (token axis non-contiguous) and the cache
                # memory is ordered (block, slot, head, cell)
                value = torch.randn(n, num_kv_heads * 128 + 64, device=DEV, generator=g).bfloat16()
                value = value[:, 64:].view(n, num_kv_heads, 128)
                slots = torch.randperm(60 * BLOCK, device=DEV, generator=g)[:n]
                slots[1::5] = -1  # tokens to skip
                cache = torch.full((60, BLOCK, num_kv_heads, codec.cell_bytes(128, mode)), 0xAB, dtype=torch.uint8,
                                   device=DEV).transpose(1, 2)
                before = cache.clone()
                kernel.write(key, value, cache, slots, k_bits=k_bits, v_bits=v_bits,
                             mean_k_rot=None if mean_k is None else kernel.rotate_mean(mean_k),
                             mean_v_rot=None if mean_v is None else kernel.rotate_mean(mean_v))
                valid = slots >= 0
                want = torch.cat([codec.pack(key[valid], k_bits, mean_k), codec.pack(value[valid], v_bits, mean_v)], -1)
                got = cache[slots[valid] // BLOCK, :, slots[valid] % BLOCK]
                untouched = torch.ones_like(cache, dtype=torch.bool)
                untouched[slots[valid] // BLOCK, :, slots[valid] % BLOCK] = False
                assert (cache == before)[untouched].all(), "wrote outside the target cells"
                same = (got == want).float().mean().item()
                rel = lambda cell, x, off, bits, mean: (
                    (codec.unpack(cell[..., off : off + codec.side_bytes(128, bits)], bits, mean) - x.float()).pow(2).mean().sqrt()
                    / x.float().pow(2).mean().sqrt()).item()
                e = [rel(c, x, off, bits, mean) for x, off, bits, mean in
                     [(key[valid], 0, k_bits, mean_k), (value[valid], k_bytes, v_bits, mean_v)] for c in (got, want)]
                print(f"  H_kv={num_kv_heads} {mode} n={n:3d}  match {same:.4f}   K {e[0]:.4f} | {e[1]:.4f}   V {e[2]:.4f} | {e[3]:.4f}")
                assert same > 0.98 and e[0] < e[1] * 1.02 + 1e-4 and e[2] < e[3] * 1.02 + 1e-4, "write kernel mismatch"


def pack_write(key, value, cache, slots, k_bits, v_bits, mean_k, mean_v) -> None:
    cell = torch.cat([codec.pack(key, k_bits, mean_k), codec.pack(value, v_bits, mean_v)], dim=-1)
    cache[slots // BLOCK, :, slots % BLOCK] = cell


def bench(fn, n=200) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


def main() -> None:
    torch.manual_seed(0)
    mixed = [(1, 1), (700, 700), (533, 1), (40, 7), (2000, 1), (17, 17), (96, 33)]
    decode = [(1, 1), (5, 1), (1700, 1), (512, 1), (33, 1)]
    print("accuracy: max error of kernel vs unpack reference / reference RMS, dot ieee | tf32 "
          "(in parentheses: relative RMS error against the K/V before quantization)")
    for num_heads, num_kv_heads in [(16, 8), (4, 4)]:
        for mode in codec.MODES:
            for name, cases, segs in [("mixed", mixed, 1), ("mixed", mixed, 16), ("decode", decode, 16), ("decode", decode, 1)]:
                b = make_batch(mode, cases, num_heads, num_kv_heads, 128)
                scale = 128**-0.5 * 0.3  # random K gives high-variance scores: shrink so the softmax does not collapse onto one key
                ref, orig = reference(b, scale, True), reference(b, scale, False)
                errs = []
                for exact, limit in [(True, 1e-4), (False, 5e-2)]:  # tf32: allow rounding error under the exaggerated K magnitudes
                    out = run_kernel(b, scale, segs, exact)
                    assert torch.isfinite(out).all(), "NaN or unwritten output"
                    errs.append(((out - ref).abs().max() / ref.pow(2).mean().sqrt()).item())
                    assert errs[-1] < limit, f"does not match the reference: {errs[-1]}"
                q_err = ((out - orig).pow(2).mean().sqrt() / orig.pow(2).mean().sqrt()).item()
                print(f"  H={num_heads}/{num_kv_heads} {mode} {name:4s} seg={segs:2d}  {errs[0]:.2e} | {errs[1]:.2e}  ({q_err:.3f})")

    check_write()

    print("speed: one decode step, one layer (us). H=16/8, D=128")
    for mode in codec.MODES:
        k_bits, v_bits = codec.MODES[mode]
        key = torch.randn(1, 8, 128, device=DEV).bfloat16()
        cache = torch.zeros(60, 8, BLOCK, codec.cell_bytes(128, mode), dtype=torch.uint8, device=DEV)
        slots = torch.tensor([77], device=DEV)
        t_pack = bench(lambda: pack_write(key, key, cache, slots, k_bits, v_bits, None, None))
        t_kernel = bench(lambda: kernel.write(key, key, cache, slots, k_bits=k_bits, v_bits=v_bits))
        print(f"  write {mode}  PyTorch pack {t_pack:5.0f}   kernel {t_kernel:4.0f}")
    for seq_len in [128, 512, 2000]:
        for mode in codec.MODES:
            b = make_batch(mode, [(seq_len, 1)], 16, 8, 128)
            scale = 128**-0.5
            t_ref = bench(lambda: reference(b, scale, True))
            t = [bench(lambda: run_kernel(b, scale, segs, exact)) for exact in (False, True) for segs in (1, 16)]
            print(f"  context {seq_len:4d} {mode}  PyTorch unpack {t_ref:5.0f}   tf32 seg=1 {t[0]:4.0f}  seg=16 {t[1]:4.0f}"
                  f"   ieee seg=1 {t[2]:4.0f}  seg=16 {t[3]:4.0f}")
    print("passed")


if __name__ == "__main__":
    main()
