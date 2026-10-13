"""KV cache codec: packs one (token, head) vector into 4 or 8 bits per value (pure PyTorch, no vLLM)

Cell layout of one side (K or V), for head_size D
    4-bit: [ D/2 data bytes | lo 2 | step 2 ]   asymmetric, 16 levels, two values per byte (even channel = low nibble)
    8-bit: [ D data bytes   | scale 2 ]         symmetric, -127..127
    lo / step / scale are bf16

Before quantizing
    1) channel-mean removal (optional): 56-90% of the energy of K is a per-channel constant unrelated to
       the token; left in, it uses up the 4-bit levels. The (head, channel) mean from calibration data
       is subtracted before storing
    2) random sign + Hadamard rotation: spreads magnitude concentrated in a few channels over all of
       them. Orthogonal, hence invertible

Modes (K bits, V bits) and cache cell size (D=128, bf16 cell 512 bytes)
    k8v4: 130 + 68 = 198 bytes (0.387x)   no calibration
    k4v4:  68 + 68 = 136 bytes (0.266x)   needs the channel-mean table
"""

from __future__ import annotations

import torch

MODES = {"k8v4": (8, 4), "k4v4": (4, 4)}
_ROT: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def side_bytes(head_size: int, bits: int) -> int:
    """Bytes of one side (K or V) of a cell"""
    return head_size // 2 + 4 if bits == 4 else head_size + 2


def cell_bytes(head_size: int, mode: str) -> int:
    """Bytes of a K + V cache cell (vLLM `state_content_bytes`)"""
    k_bits, v_bits = MODES[mode]
    return side_bytes(head_size, k_bits) + side_bytes(head_size, v_bits)


def rotation(head_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(sign (D,), hadamard (D, D)). hadamard is symmetric and orthogonal"""
    key = (head_size, str(device))
    if key not in _ROT:
        assert head_size & (head_size - 1) == 0, "head_size must be a power of two"
        h = torch.ones(1, 1)
        while h.shape[0] < head_size:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        sign = torch.randint(0, 2, (head_size,), generator=torch.Generator().manual_seed(0)).float() * 2 - 1
        _ROT[key] = (sign.to(device), (h / head_size**0.5).to(device))
    return _ROT[key]


def pack(x: torch.Tensor, bits: int, mean: torch.Tensor | None = None) -> torch.Tensor:
    """(..., D) float -> (..., side_bytes) uint8. mean: channel mean broadcastable to x"""
    x = x.float()
    if mean is not None:
        x = x - mean
    sign, had = rotation(x.shape[-1], x.device)
    x = (x * sign) @ had
    if bits == 4:
        lo = x.amin(dim=-1, keepdim=True).to(torch.bfloat16)
        step = ((x.amax(dim=-1, keepdim=True) - lo.float()) / 15).clamp_min(1e-6).to(torch.bfloat16)
        q = ((x - lo.float()) / step.float()).round().clamp_(0, 15).to(torch.uint8)
        data = q[..., 0::2] | (q[..., 1::2] << 4)
        return torch.cat([data, lo.view(torch.uint8), step.view(torch.uint8)], dim=-1)
    scale = (x.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-6).to(torch.bfloat16)
    q = (x / scale.float()).round().clamp_(-127, 127).to(torch.int8)
    return torch.cat([q.view(torch.uint8), scale.view(torch.uint8)], dim=-1)


def unpack(cell: torch.Tensor, bits: int, mean: torch.Tensor | None = None,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(..., side_bytes) uint8 -> (..., D) `dtype`"""
    if bits == 4:
        n = cell.shape[-1] - 4
        data = cell[..., :n]
        q = torch.stack([data & 0xF, data >> 4], dim=-1).flatten(-2).float()  # restore the channel order
        lo = cell[..., n : n + 2].contiguous().view(torch.bfloat16).float()
        step = cell[..., n + 2 :].contiguous().view(torch.bfloat16).float()
        x = q * step + lo
    else:
        n = cell.shape[-1] - 2
        q = cell[..., :n].contiguous().view(torch.int8).float()
        x = q * cell[..., n:].contiguous().view(torch.bfloat16).float()
    sign, had = rotation(x.shape[-1], x.device)
    x = (x @ had) * sign
    if mean is not None:
        x = x + mean
    return x.to(dtype)
