"""vLLM plugin: registers our KV cache codecs as the CUSTOM attention backend

- `register` is called through the `vllm.general_plugins` entry point group (pyproject.toml)
- start a server: `KVCODEC=k8v4 vllm serve <model> --attention-backend CUSTOM`
- KVCODEC
    k8v4 (default)  K 8-bit + V 4-bit, cache 0.387x                       kvcodec.backend
    k4v4            K and V 4-bit + channel-mean removal, cache 0.266x    kvcodec.backend (needs KVCODEC_MEAN)
    fft             FFT along the sequence axis, compress then restore    kvcodec.fft_backend
                    in place (quality check only, no memory saving, needs `--block-size 256`)
"""

import os

_FFT = "kvcodec.fft_backend.FFTKVAttentionBackend"
_CODEC = "kvcodec.backend.CodecAttentionBackend"


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    path = _FFT if os.environ.get("KVCODEC", "k8v4") == "fft" else _CODEC
    register_backend(AttentionBackendEnum.CUSTOM, path)
