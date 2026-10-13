"""KV 캐시 코덱: (토큰, head) 벡터 하나를 4비트 또는 8비트로 압축 (순수 PyTorch, vLLM 의존 없음)

한쪽(K 또는 V)의 칸 형식, head_size D 기준
    4비트: [ 값 D/2 바이트 | lo 2 | step 2 ]   비대칭 16단계, 값 두 개를 한 바이트에 (짝수 채널 = 하위 4비트)
    8비트: [ 값 D 바이트   | scale 2 ]         대칭 -127..127
    lo / step / scale은 bf16

양자화 전 처리
    1) 채널 평균 제거 (선택): K는 에너지의 56~90%가 토큰과 무관한 채널별 상수라, 그대로 두면 4비트
       눈금이 그 상수를 표현하는 데 소진됨. 보정 데이터로 구한 (head, 채널) 평균을 빼고 저장
    2) 무작위 부호 + 아다마르 회전: 소수 채널에 몰린 크기를 전 채널에 분산. 직교 변환이라 역변환 가능

모드 (K 비트, V 비트)와 캐시 칸 크기 (D=128, bf16 원본 512바이트)
    k8v4: 130 + 68 = 198바이트 (0.387x)   보정 불필요
    k4v4:  68 + 68 = 136바이트 (0.266x)   채널 평균 표 필요
"""

from __future__ import annotations

import torch

MODES = {"k8v4": (8, 4), "k4v4": (4, 4)}
_ROT: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def side_bytes(head_size: int, bits: int) -> int:
    """K 또는 V 한쪽의 칸 바이트 수"""
    return head_size // 2 + 4 if bits == 4 else head_size + 2


def cell_bytes(head_size: int, mode: str) -> int:
    """K + V 캐시 칸 바이트 수 (vLLM `state_content_bytes`)"""
    k_bits, v_bits = MODES[mode]
    return side_bytes(head_size, k_bits) + side_bytes(head_size, v_bits)


def rotation(head_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(sign (D,), hadamard (D, D)). hadamard는 대칭 직교 행렬"""
    key = (head_size, str(device))
    if key not in _ROT:
        assert head_size & (head_size - 1) == 0, "head_size는 2의 거듭제곱"
        h = torch.ones(1, 1)
        while h.shape[0] < head_size:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        sign = torch.randint(0, 2, (head_size,), generator=torch.Generator().manual_seed(0)).float() * 2 - 1
        _ROT[key] = (sign.to(device), (h / head_size**0.5).to(device))
    return _ROT[key]


def pack(x: torch.Tensor, bits: int, mean: torch.Tensor | None = None) -> torch.Tensor:
    """(..., D) 실수 -> (..., side_bytes) uint8. mean: x에 broadcast 가능한 채널 평균"""
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
        q = torch.stack([data & 0xF, data >> 4], dim=-1).flatten(-2).float()  # 채널 순서 복원
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
