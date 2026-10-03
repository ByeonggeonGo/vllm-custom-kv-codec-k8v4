# transformers-generate-optimization

Memory and inference-speed optimization of PyTorch/Transformers `generate()`, measured
against vLLM.

`model.generate(output_logits=True)` keeps one full-vocabulary logit row per decode step
and returns all of them, so any per-step analysis can only happen after the loop ends.
Writing the decode loop out by hand and reducing each step in place removes that term.
This measures what the two cost, and puts vLLM beside them for scale.

Measured on an RTX 5080 (16 GB) with Qwen3-4B-FP8, greedy decoding, 31-token prompt.
Another process held 4218 MiB of the card throughout.

## Speed — tokens/sec

| max_tokens | vLLM | `generate()` | manual loop |
| ---------: | ---: | -----------: | ----------: |
|         32 | **154.9** | 49.8 | 49.9 |
|        128 | **157.1** | 49.4 | 50.0 |
|        512 | **157.6** | 49.3 | 49.7 |

vLLM is ~3.2x faster. `generate()` and the hand-written loop are indistinguishable: the
cost is in the forward pass, not in the Python loop around it.

## Decode memory — MiB above the weights

| max_tokens | vLLM | `generate()` | manual loop |
| ---------: | ---: | -----------: | ----------: |
|         32 |  0.0 | 29.0 | **15.7** |
|        128 |  0.0 | 98.1 | **24.8** |
|        512 |  0.0 | 406.6 | **125.5** |

The gap divided by step count converges on the size of one float32 logit row
(`151936 x 4 B = 0.580 MiB`):

| max_tokens | extra MiB | per step |
| ---------: | --------: | -------: |
|         32 |      13.3 |    0.417 |
|        128 |      73.3 |    0.573 |
|        512 |     281.1 |    0.549 |

At 512 steps that retained-logits term is 69% of what decoding cost `generate()`.

## Two things the numbers correct

**The manual loop is not constant-memory.** It grows 15.7 -> 24.8 -> 125.5 MiB, because
the KV cache grows with sequence length no matter how the loop is written. Both paths pay
that term linearly; `generate()` pays a second linear term on top. Only the second one is
removable.

| max_tokens | seq len | manual decode | of which KV cache* |
| ---------: | ------: | ------------: | -----------------: |
|         32 |      63 |          15.7 |                8.9 |
|        128 |     159 |          24.8 |               22.4 |
|        512 |     543 |         125.5 |               76.4 |

\*computed from the architecture (36 layers, 8 KV heads, head_dim 128, bf16), not measured.

**vLLM's 0.0 is not a win.** Its server claims `--gpu-memory-utilization` of the card at
startup — 9860 MiB here, flat across every budget — and works inside that slab. The
per-request delta is zero because the cost was paid up front, not because it is absent.
vLLM's memory figure is also an `nvidia-smi` sample of another process, not a torch
allocator peak, so it is not on the same axis. The comparison vLLM genuinely contributes
here is speed.

## Running it

Each script is standalone — no shared module, no package to install from this repo.

```
# with a vLLM server on localhost:8000 serving as "local"
python run_vllm.py

# with every vLLM server stopped (it reserves VRAM at startup, so idle is not enough)
python run_generate.py
python run_manual.py
```

Each writes `results/<path>.json` after every budget, so an OOM on a long run does not
discard the budgets that already succeeded. Then open `compare.ipynb`; it skips any path
that has no result file yet.

The scripts print how much GPU memory other processes hold and warn above 500 MiB. A
shared card skews the peaks and can OOM a long run outright — the figure is recorded in
each result file's `meta` so a surprising measurement can be explained later.

## Requirements

```
pip install -r requirements.txt
```

CUDA is required. The FP8 checkpoint needs Triton kernels, which need Python headers
(`sudo apt install python3.12-dev`) and the `kernels` package; a bf16 model such as
`Qwen/Qwen3-0.6B` avoids the FP8 path but still uses Triton on recent torch builds.
Change `MODEL` at the top of each script to switch.
