# fft-based-kv-memory-compression

Custom KV-cache codecs for vLLM, shipped as a plugin, with a web bench that runs them next to
vLLM's built-in cache formats and plots memory and timing live.

The repo started as an attempt to compress the KV cache with FFT. That failed (results folded
below) and pointed at the real problem — the distribution of the key vectors — which the codecs
here address directly.

![KV cache codec bench](docs/demo.png)

## Result

Qwen3-0.6B on vLLM 0.30.0, one RTX 5080. Every server gets the same 256 MB KV cache; only the
cache format differs.

| Server | Cache cell | Tokens that fit in 256 MB | KV used, 1,695-token prompt + answer | PPL | Decode speed | Time to first token |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (vLLM default) | 512 B (1.000x) | 2,336 | 191 MB | 30.14 | 188 tok/s | 29 ms |
| vLLM `int8_per_token_head` | 264 B (0.516x) | 4,528 | 98 MB | 31.04 | 190 tok/s | 38 ms |
| **ours `k8v4`** | **198 B (0.387x)** | **6,048** | **74 MB** | **30.44** | **195 tok/s** | 52 ms |
| vLLM `int4_per_token_head` | 136 B (0.266x) | 8,800 | 50 MB | 83.61 | 113 tok/s | 45 ms |
| **ours `k4v4`** | **136 B (0.266x)** | **8,800** | **51 MB** | **33.61** | **191 tok/s** | 55 ms |

- **Memory is really smaller.** The plugin declares a smaller cache cell, vLLM allocates the pool
  with it, and vLLM's own `kv_cache_usage_perc` metric reports the difference: the 1,695-token
  prompt occupies 74.5% of the bf16 pool, 28.9% with `k8v4`, 19.9% with `k4v4`.
- **Quality is better than the built-ins at the same size.** `k8v4` is 25% smaller than vLLM int8
  and closer to bf16. `k4v4` is the size of vLLM int4 with perplexity 33.6 instead of 83.6.
- **Decoding is as fast as bf16.** Triton kernels ([`kvcodec/kernel.py`](kvcodec/kernel.py)) read
  the packed cells without unpacking them and write new tokens. With the first version, which
  unpacked the whole context in PyTorch on every step, `k8v4` decoded at 57 tok/s.
- **Prefill is slower.** Time to first token on a 1,682-token prompt is 52–55 ms against 29 ms for
  bf16.

All five rows come from [`demo/measure.py`](demo/measure.py). PPL is measured inside vLLM the same
way on every server: token log-probabilities of six ~1,500-token passages (9,482 tokens) from
`/v1/completions` with `echo`. Speed is tokens per second after the first token on a 1,682-token
prompt, best of three, one request at a time, all servers eager with prefix caching off.

## How the codecs work

A cache cell holds the key and the value of one (token, head): 2 × 128 bf16 values = 512 bytes.

```
k8v4   [ K: 128 × int8 | scale ] [ V: 64 bytes of packed int4 | lo | step ]   = 130 + 68 = 198 B
k4v4   [ K:  64 bytes of packed int4 | lo | step ] [ V: same ]              =  68 + 68 = 136 B
```

Three things make 4 bits usable ([`kvcodec/codec.py`](kvcodec/codec.py)):

1. **Rotation.** A key vector has a few channels 100–260× larger than its median. A scale fitted to
   the maximum rounds 70–95% of the other channels to zero. Multiplying by a random-sign Hadamard
   matrix spreads the magnitude over all 128 channels. The matrix is orthogonal, so the read path
   rotates the query instead of undoing it on every key.
2. **Channel-mean removal** (`k4v4`). 56–90% of the energy of the keys is a per-channel constant
   that does not depend on the token. Left in, it uses up the 16 levels. A table of per-(layer,
   head, channel) means, computed once by [`kvcodec/calibrate.py`](kvcodec/calibrate.py), is
   subtracted before quantizing.
3. **Spend bits where they matter** (`k8v4`). Values tolerate 4 bits (KL 0.017 against bf16); keys
   do not (KL 1.42). Keys get 8 bits, values 4.

| Step, both sides 4-bit | PPL (bf16 33.7) |
|---|---:|
| per-(token, head) scale only | 37,146 |
| + rotation | 127 |
| + asymmetric 16 levels | 82 |
| + channel-mean removal (`k4v4`) | 39 |
| keys 8-bit, values 4-bit, rotation (`k8v4`) | 34 |

## How it plugs into vLLM

vLLM is not modified or vendored. The package registers itself through the `vllm.general_plugins`
entry point; vLLM imports it at startup and uses it when started with `--attention-backend CUSTOM`.

```mermaid
flowchart LR
    P["pip install .<br/>(entry point: vllm.general_plugins)"] --> R["vLLM startup<br/>load_general_plugins()"]
    R --> B["register_backend(CUSTOM,<br/>kvcodec.backend.CodecAttentionBackend)"]
    B --> S["customize_spec()<br/>cell = 198 or 136 bytes"]
    S --> A["vLLM allocates the pool<br/>with that cell size"]
    A --> W["do_kv_cache_update()<br/>pack new tokens into pages"]
    W --> F["forward()<br/>attention on packed pages"]
```

[`kvcodec/backend.py`](kvcodec/backend.py) subclasses vLLM's Triton backend and overrides three things:
`customize_spec` (cell size), `do_kv_cache_update` (write) and `forward` (read).

## The kernels

[`kvcodec/kernel.py`](kvcodec/kernel.py) never reconstructs a key or a value. With `c` the stored
integer codes, `R` the rotation and `q_rot = R q`:

```
score    q · k  =  q_rot · k_rot + q · mean_k
                =  step · (q_rot · c) + lo · sum(q_rot)  + const     one dot product with the codes
output   Σ p_t v_t  =  Rᵀ (Σ p_t v_rot_t) + mean_v                   accumulate rotated, undo once
```

- `q · mean_k` is the same for every key of a query, so it cancels in the softmax: the key mean
  table is only needed when writing.
- The query is rotated once per step and the output un-rotated once, whatever the context length.
- Packed int4 is read as low nibbles then high nibbles; the channel order this implies is folded
  into the two rotation matrices, so the kernel does no interleaving.
- Decoding has one query per sequence and would use 8 GPU programs. The context is split into 16
  segments computed in parallel, and a second small kernel merges their softmax states.
- The write kernel rotates, quantizes and packs new tokens; its output is byte-identical to
  `codec.pack`.

| One decode step, one layer, 2,000-token context | PyTorch (v1) | Triton |
|---|---:|---:|
| read `k8v4` | 225 µs | 47 µs |
| write `k8v4` | 143 µs | 10 µs |

`python -m kvcodec.kernel_check` checks both kernels against the PyTorch codec (mixed prefill and
decode batches, both modes) and prints these timings. `KVCODEC_READ=torch` / `KVCODEC_WRITE=torch`
switch the server back to the PyTorch paths.

## Running it

```bash
# five vLLM servers in docker, same model, same KV cache budget (needs ~11 GB of GPU memory)
demo/servers.sh up            # or: demo/servers.sh up bf16 ours-k8v4
python demo/app.py            # http://localhost:8200  (standard library only)
demo/servers.sh down
```

The page sends one prompt to the selected servers, streams the answers side by side, and plots
KV cache in use and GPU memory over time, and KV memory, decode speed and time to first token
against prompt length. "Sweep context length" runs the prompt at six context sizes to draw the trend.

To use a codec in your own vLLM 0.30 environment:

```bash
pip install --no-deps .
KVCODEC=k8v4 vllm serve Qwen/Qwen3-0.6B --attention-backend CUSTOM --enforce-eager
KVCODEC=k4v4 KVCODEC_MEAN=results/kv_mean_qwen3-0.6b.pt vllm serve Qwen/Qwen3-0.6B --attention-backend CUSTOM --enforce-eager
```

| Path | Contents |
|---|---|
| `kvcodec/codec.py` | pack / unpack, rotation, cell sizes (pure PyTorch) |
| `kvcodec/kernel.py` | Triton kernels: attention on packed cells, write |
| `kvcodec/kernel_check.py` | kernel correctness and timing against the PyTorch codec |
| `kvcodec/backend.py` | vLLM attention backend for `k8v4` / `k4v4` |
| `kvcodec/calibrate.py` | channel-mean table for `k4v4` |
| `kvcodec/fft_backend.py` | the FFT codec as a backend (quality only, no memory saving) |
| `demo/servers.sh`, `demo/app.py`, `demo/index.html` | servers and the web bench |
| `demo/measure.py` | perplexity, decode speed and time to first token for the table above |

## Limits

- Tested on vLLM 0.30.0 only. `customize_spec` is marked a temporary API in vLLM's source.
- Eager mode, prefix caching off. CUDA graphs and prefix caching are untested with the plugin.
- The kernels implement causal attention only: no sliding window, ALiBi, attention sinks or soft
  cap. Dot products use TF32 tensor cores like vLLM's kernels (`KVCODEC_DOT=ieee` for exact
  float32, about 2.7× slower prefill).
- Prefill is about 1.8× slower than bf16.
- `k4v4` needs a calibration table per model; the keys are post-RoPE, so the table should be built
  with windows at least as long as the contexts served.
- One 0.6B model, one book for perplexity. Both 4-bit formats (vLLM's and this one) break on very
  short prompts.

<details>
<summary><b>Earlier experiment: FFT compression of the KV cache (negative result)</b></summary>

Compressing a transformer's KV cache by storing a few FFT coefficients per channel instead of
the raw values, measured on Qwen3-0.6B against plain KV quantization and context truncation.

**Result: negative.** On this benchmark, FFT compression of the KV cache loses to the two
simplest baselines.

- Plain int4 KV quantization is near-lossless at **0.281x** (perplexity 27.7 vs 27.6 uncompressed).
- Keeping only the last 64 tokens, uncompressed, gives perplexity **32.7 at 0.125x**.
- The best FFT setting — sequence-axis FFT, a learned coefficient selector, int4 coefficients —
  gives **39.9 at 0.111x** and stays at 40–45 all the way down to 0.02x. Discarding the past
  entirely gives 46.2. The compressed past carries little usable information; training the
  selector mostly teaches it to make the damaged cache harmless.
- The original plan, FFT along `head_dim`, is far worse: perplexity 261,328 at half the coefficients.

![Cache size vs quality](docs/fig_pareto.png)

| Setting | Stored bytes | KL | Top-1 agree | PPL |
|---|---:|---:|---:|---:|
| bf16 KV, full context | 1.000x | 0 | 1.000 | 27.56 |
| **past discarded entirely** | 0x | 0.66 | 0.651 | 46.20 |
| keep last 8 tokens | 0.016x | 0.47 | 0.680 | 39.15 |
| keep last 64 tokens | 0.125x | 0.24 | 0.771 | 32.71 |
| keep last 256 tokens | 0.500x | 0.09 | 0.857 | 29.04 |
| plain int8 KV | 0.531x | 0.001 | 0.981 | 27.56 |
| plain int4 KV | 0.281x | 0.007 | 0.955 | 27.70 |
| plain int2 KV, group 32 | 0.188x | 0.11 | 0.816 | 30.06 |
| plain int2 KV, group 128 | 0.141x | 0.34 | 0.688 | 36.24 |
| plain int1 KV, group 32 | 0.125x | 6.95 | 0.082 | 19,069 |
| `head_dim`-axis FFT, energy top-50% | 0.535x | 9.38 | 0.040 | 261,328 |
| sequence-axis FFT, energy top-50%, block 256 | 0.533x | 1.41 | 0.469 | 124.5 |
| sequence-axis FFT, learned top-50% | 0.533x | 0.40 | 0.657 | 35.6 |
| learned top-50% + int8 coefficients | 0.299x | 0.40 | 0.655 | 35.6 |
| learned top-25% + int4 coefficients | 0.111x | 0.55 | 0.624 | 39.9 |
| learned top-12.5% + int4 coefficients | 0.070x | 0.61 | 0.607 | 41.4 |
| learned top-3.1% + int4 coefficients | 0.020x | 0.64 | 0.593 | 42.0 |

Setup: Qwen3-0.6B (bf16, weights frozen), 640-token windows from a Gutenberg novel. The first
512 tokens are compressed in the cache; quality is measured on the next 128 tokens
(16 batches × 2 sequences = 4,096 predictions). KL is KL(teacher ‖ student) per token.
"Keep last N tokens" re-runs the model on the last N past tokens plus the evaluated ones.
"Stored bytes" is counted from the tensors the codec actually keeps — coefficients,
quantization scales and the selection bitmask — divided by the bf16 cache.

### How it works

```mermaid
flowchart LR
    A["KV cache<br/>(B, H, S, 128)"] --> B["split into<br/>256-token blocks"]
    B --> C["rFFT along tokens<br/>per (head, channel)<br/>129 coefficients"]
    C --> D["selector picks top-k<br/>log|X| + learned bias"]
    D --> E["quantize kept coefs<br/>int8 / int4,<br/>scale per 8–16 coefs"]
    E --> F[("stored:<br/>coefs + scales<br/>+ 129-bit mask")]
    F -->|read| G["scatter + irFFT"]
    G --> H["attention"]
```

![FFT axis](docs/fig_axes.png)

1. **Sequence-axis block rFFT** (`seq_codec`). The cache is cut into 256-token blocks and each
   (head, channel) series of 256 values is transformed. Only the selected coefficients, their
   quantization scales and a 129-bit selection mask are kept; reads scatter them back and irFFT.
2. **Learned selector** (`SeqFreqSelector`). Score = `log|X| + bias[layer, K/V, head, freq, channel]`,
   keep the top-k. The bias starts at zero, so training starts exactly at energy top-k and any
   gain is the selector's. Top-k is trained through a straight-through estimator on the KL of the
   tokens that read the compressed cache. 7.4M parameters (1.2% of the model), ~1 minute per setting.
3. **Grouped coefficient quantization** (`coef_group`). The DC term is tens of times larger than
   the rest of a block's spectrum, so one scale per block crushes the small coefficients to zero.
   A scale per 8–16 coefficients fixes it.
4. **Byte accounting by construction.** The codec builds the storage format and reconstructs from
   it alone, so the quality numbers and the byte numbers come from the same tensors.

#### Evaluation without future leakage

A sequence-axis filter mixes information from later tokens of a block into earlier ones. Applied
inside a single forward pass, a query would see the future. Compression is therefore applied the
way a KV cache is used: fill, compress, then decode on top of it.

```mermaid
flowchart LR
    P["tokens 0–511"] -->|prefill| C1["KV cache"]
    C1 -->|compress + reconstruct| C2["compressed cache"]
    T["tokens 512–639"] -->|forward on top of| C2
    C2 --> M["KL, top-1 agreement, PPL<br/>vs. uncompressed run"]
```

The head-axis method is re-measured with the same protocol, so every row of the table above is
comparable.

### What moved the numbers

![Each step at 50% of coefficients](docs/fig_steps.png)

At a fixed budget of half the coefficients:

| Step | KL | PPL | vs. previous |
|---|---:|---:|---|
| `head_dim`-axis FFT, energy top-k | 9.38 | 261,328 | – |
| → sequence axis, block 64 | 2.15 | 276.6 | KL ÷4.4 |
| → block 256 | 1.41 | 124.5 | KL ÷1.5 |
| → learned selector | 0.40 | 35.6 | KL ÷3.5 |
| → int8 coefficients (bytes 0.533x → 0.299x) | 0.40 | 35.6 | same quality, 44% fewer bytes |

Each step is a real improvement over the one before, but the end point (KL 0.40) is still
closer to "no past at all" (KL 0.66) than to the full context. Untrained energy top-k (KL 1.41)
is *worse* than having no past: a damaged cache misleads attention more than a missing one.

#### Why the sequence axis

![Spectrum sparsity](docs/fig_spectrum.png)

The share of energy held by the top half of the FFT coefficients, sorted per sample. Sorting
makes even white noise score 0.84, so that, not 0.5, is the line to beat. Along `head_dim` the
K cache sits below noise (0.79) and V on it (0.84): there is nothing to compress. Along the
sequence the K cache reaches 0.89 even with the DC term removed.

### Findings

- **`head_dim`-axis FFT has no structure to exploit.** Dropping 12.5% of the coefficients already
  gives KL 4.06; dropping half gives perplexity 261,328.
- **Which coefficients are kept matters more than how many.** At 75% kept, sequence-axis KL is
  0.30 for energy top-k, 1.52 for a fixed low-pass and 2.48 for random. Low-pass barely beats
  random, so "K/V varies slowly over time" does not explain the result.
- **Longer blocks help.** KL at 50% kept: 2.70 / 2.15 / 1.75 / 1.41 for blocks of 32 / 64 / 128 / 256.
  The cost is that up to 255 recent tokens stay uncompressed.
- **RoPE is not the problem.** The K spectrum shows a RoPE peak (bin ≈ 41), but undoing RoPE before
  compressing leaves KL unchanged (0.304 → 0.306).
- **Separate K/V budgets do not pay.** K is more sensitive than V, and K-heavy splits beat V-heavy
  ones (KL 0.73 vs 1.00 at 0.662x), but an even split at the same size is better still (0.59 at 0.666x).
- **Learning helps most where compression is hardest.** Energy → learned KL: 0.20 → 0.16 at 75% kept,
  1.41 → 0.40 at 50%, 4.35 → 0.54 at 25% (perplexity 2,210 → 39.6). Energy-optimal coefficients are
  not the ones attention needs.
- **int8 coefficients are free; int4 needs grouped scales.** int8 keeps KL at 0.40 while bytes drop
  0.533x → 0.299x. int4 at 100% kept: KL 3.26 with one scale per block, 0.65 per 32 coefficients,
  0.12 per 8.
- **Plain quantization is the bar above 0.14x.** int4 KV is near-lossless at 0.281x (KL 0.007).
  int2 with group 32 reaches perplexity 30.1 at 0.188x, against 36.9 for the FFT route at 0.189x.
  Dropping coefficients costs more than rounding them. Below 2 bits it stops working: int1 gives
  perplexity above 19,000.
- **Truncation is the bar below 0.14x, and FFT does not clear it.** Keeping the last 64 tokens
  (0.125x) gives perplexity 32.7; the last 8 tokens (0.016x) give 39.2. The FFT route gives 39.9 at
  0.111x and 42.0 at 0.020x.
- **The FFT curve is flat because the compressed past is barely used.** From 0.28x to 0.02x its KL
  only moves from 0.53 to 0.64, next to the 0.66 of discarding the past. A method that keeps
  information would degrade as bytes are removed.
- **This benchmark has little long-range signal.** Full context is worth perplexity 27.6 vs 46.2
  with none, and the last 64 tokens already recover most of it. A task that needs the far past
  (retrieval, long-document QA) would separate the methods more sharply.

### Limitations

- One 0.6B model, one book; 4,096 evaluated tokens. Next-token perplexity on a novel rewards
  recent context, so it understates what a faithful cache is worth.
- A single 512-token compressed span. Long contexts and block-by-block updates during generation
  are not measured.
- No speed measurement. Every block adds an FFT on write and an irFFT on read.
- `SparseSpectralCache` (section 7 of the notebook) is the `head_dim`-axis prototype and ignores
  `active_layers`; a sequence-axis cache for `generate()` is not written yet.
- Training is not bit-reproducible on GPU; KL moves by about ±0.01 between runs.

### Next

- **Evaluate on a task that needs the far past** (passkey retrieval, long-document QA). On such a
  task truncation fails outright, which is the only regime where a lossy codec can earn its bytes.
- **FFT on the quantization residual.** Store plain int2 and correct only its residual with
  sequence-axis FFT, instead of replacing the values with coefficients.
- **Per-frequency bit allocation** (transform coding with a DCT) instead of keep-or-drop.

### Reproducing

```
pip install -r requirements.txt
jupyter notebook fft_kv_compression.ipynb   # sections 0–1 (setup), 2 (spectrum), 9–10
```

Section 9 runs in a few minutes on one GPU (RTX 5080 here). Sections 2–8 hold the earlier
`head_dim`-axis experiments, including selector training for that axis.

| File | Contents |
|---|---|
| `custom_layers.py` | `seq_codec`, `raw_codec`, `head_codec`, `SeqFreqSelector`, `seq_filter_ste`; head-axis `KVFreqCompressor`, `FFTCompressedCache`, `SparseSpectralCache` |
| `tools.py` | model and data loading, `evaluate_prefix`, `compress_cache_`, `train_seq_selector`, spectrum statistics |
| `fft_kv_compression.ipynb` | all experiments and the conclusion (Korean) |
| `results/fft_kv_summary.json` | every measurement behind the tables |

</details>

<details>
<summary><b>Background: where the KV cache shows up in `generate()`</b></summary>

This repo started as a memory and speed study of `model.generate()`, which is what pointed at
the KV cache in the first place.

Memory and inference-speed optimization of PyTorch/Transformers `generate()`, measured
against vLLM.

`model.generate(output_logits=True)` keeps one full-vocabulary logit row per decode step
and returns all of them, so any per-step analysis can only happen after the loop ends.
Writing the decode loop out by hand and reducing each step in place removes that term.
This measures what the two cost, and puts vLLM beside them for scale.

Measured on an RTX 5080 (16 GB) with Qwen3-4B-FP8, greedy decoding, 31-token prompt.
Another process held 4218 MiB of the card throughout.

#### Speed — tokens/sec

| max_tokens | vLLM | `generate()` | manual loop |
| ---------: | ---: | -----------: | ----------: |
|         32 | **154.9** | 49.8 | 49.9 |
|        128 | **157.1** | 49.4 | 50.0 |
|        512 | **157.6** | 49.3 | 49.7 |

vLLM is ~3.2x faster. `generate()` and the hand-written loop are indistinguishable: the
cost is in the forward pass, not in the Python loop around it.

#### Decode memory — MiB above the weights

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

#### Two things the numbers correct

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

#### Running it

Each script is standalone — no shared module, no package to install from this repo.

```
# with a vLLM server on localhost:8000 serving as "local"
python run_vllm.py

# with every vLLM server stopped (it reserves VRAM at startup, so idle is not enough)
python run_generate.py
python run_manual.py
```

Each writes `results/<path>.json` after every budget, so an OOM on a long run does not
discard the budgets that already succeeded.

The scripts print how much GPU memory other processes hold and warn above 500 MiB. A
shared card skews the peaks and can OOM a long run outright — the figure is recorded in
each result file's `meta` so a surprising measurement can be explained later.

#### Requirements

```
pip install -r requirements.txt
```

CUDA is required. The FP8 checkpoint needs Triton kernels, which need Python headers
(`sudo apt install python3.12-dev`) and the `kernels` package; a bf16 model such as
`Qwen/Qwen3-0.6B` avoids the FP8 path but still uses Triton on recent torch builds.
Change `MODEL` at the top of each script to switch.

</details>
