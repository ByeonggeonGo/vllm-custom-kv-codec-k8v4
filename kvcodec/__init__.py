"""vLLM 플러그인: 직접 구현한 KV 캐시 코덱을 CUSTOM attention 백엔드로 등록

- entry point 그룹 `vllm.general_plugins`에서 `register` 호출 (pyproject.toml)
- 서버 실행: `KVCODEC=k8v4 vllm serve <model> --attention-backend CUSTOM`
- KVCODEC
    k8v4 (기본)  K 8비트 + V 4비트, 캐시 0.387x              kvcodec.backend
    k4v4         K·V 4비트 + 채널 평균 제거, 캐시 0.266x     kvcodec.backend (KVCODEC_MEAN 필요)
    fft          sequence 축 FFT, 압축 후 복원해 제자리 기록  kvcodec.fft_backend
                 (품질 확인용, 메모리 절감 없음, `--block-size 256` 필요)
"""

import os

_FFT = "kvcodec.fft_backend.FFTKVAttentionBackend"
_CODEC = "kvcodec.backend.CodecAttentionBackend"


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    path = _FFT if os.environ.get("KVCODEC", "k8v4") == "fft" else _CODEC
    register_backend(AttentionBackendEnum.CUSTOM, path)
