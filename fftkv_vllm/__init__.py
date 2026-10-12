"""vLLM 플러그인: sequence 축 FFT KV 코덱을 CUSTOM attention 백엔드로 등록

- entry point 그룹 `vllm.general_plugins`에서 `register`가 호출됨 (pyproject.toml)
- 서버 실행: `vllm serve <model> --attention-backend CUSTOM --block-size 256`
- 코덱 설정은 환경변수 (`fftkv_vllm/backend.py`의 `CodecConfig`)
"""


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, "fftkv_vllm.backend.FFTKVAttentionBackend")
