# Qwen3.8-Flash-Next on 3× CMP170HX (SM80) with vLLM

A working recipe for serving [Intel's INT4-weight AutoRound quant] (Qwen4Exp, 180B-A6B MoE) on **3× NVIDIA CMP170HX (SM80, unlocked to 64 GiB each)** with:


| Capability | Status |
|---|---|
| W4A16 (AutoRound/INC) experts | ✅ GPTQ-Marlin, dequant verified bit-exact |
| 95.4 GiB BF16 PLE n-gram table | ✅ streamed from NVMe (never fully loaded to RAM/VRAM) |
| FP8 e4m3 KV cache | ✅ (software e4m3 on SM80) |
| Mamba SSM state | ✅ bf16 (`--mamba-ssm-cache-dtype bfloat16`) |
| Context length | ✅ 1,048,576 tokens via YaRN ×4 |
| KV pool | **5.97M tokens** (5.69× concurrent 1M sessions) |
| Fixed chat templates | [froggeric's extensively bugfixed Qwen3.8 chat templates] | 
| Decode | ~74 tok/s short-context, ~60 tok/s at 1M |
| Retrieval QA | exact needle recall at all depths through 1M |

The vLLM changes live in the fork [`clee/vllm-sm80`] (see [Patches](#patches)). This repo holds the walkthrough, launcher, serving config, and validation harness.

## Hardware notes (my specific machine)

- **No GPU P2P.** For me, CMP170HX pairs on different root buses never complete real peer transfers (BAR1 maps are created, DMA hangs). Always `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`; PP stages through host RAM via NCCL SHM, which works fine since there's very little data that actually travels between cards.
- **Pinned host memory ceiling ≈ 64 GiB.** My Threadripper machine only has 128GB of RAM, so the PLE table (95.4 GiB) can't really be pinned — that's why the NVMe path below exists.
- **No FP8 Triton pointers on SM80.** FP8 KV cache is stored as `uint8` and software-decoded (see `q38fn/fp8-kv-sm80`).

## Quick start

```bash
# 1. checkpoint on a local NVMe (needs at least 169 GiB free space)
#    hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir /home/models/qwen38fn-w4a16-autoround

# 2. pull the pinned nightly the patches are based on
docker pull vllm/vllm-openai:nightly-0cbac6cd1305f710e12193596b27488397bcb205

# 3. clone the fork (branches: q38fn/*)
git clone https://github.com/clee/vllm-sm80

# 4. build the fork into an image (the patches are pure Python; the fork's
#    vllm/ tree syncs over the pinned nightly — ~2 min) and launch:
docker build -t vllm-sm80:q38fn .
./run-qwen38fn.sh IMAGE=vllm-sm80:q38fn
```

`run-qwen38fn.sh` launches the server on :8080 (default IMAGE is the stock pinned nightly with the fork's patched files bind-mounted — handy for iteration; `IMAGE=vllm-sm80:q38fn` uses the baked image instead). The essential environment:

```text
NCCL_P2P_DISABLE=1                      # mandatory on this hardware
VLLM_PP_LAYER_PARTITION=17,17,14        # PLE layer (idx 1) stays on rank 0
VLLM_PLE_STAGED=1                       # NVMe PLE (shard-direct preadv)
QWEN4EXP_PLE_MODEL_DIR=/models/qwen38fn-w4a16-autoround
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1         # 1M > checkpoint's 262144 native
```

## Patches

Stacked on `q38fn/base` = `0cbac6cd1305` (v0.30.1rc1.dev558):

| Bookmark | What it does |
|---|---|
| `q38fn/units-fix` | Fixes a 3-instance units bug in `_align_hybrid_block_size`: one attention layer's page was compared against the all-layers mamba aggregate — spurious block doubling under fp8 KV, assert failure, and negative mamba padding. |
| `q38fn/fp8-kv-sm80` | fp8 e4m3 KV on SM80: uint8 cache storage + software decode in the QSA gather kernel, bit-level RNE encoder for writes. |
| `q38fn/ple-pp3` | PLE checkpoints under PP>1: gate removal, per-worker `set_current_vllm_config` at load, rank-unowned tensor skip in `allocate_kv_cache`, MTP draft on last rank. |
| `q38fn/ple-nvme-staged` | The 95.4 GiB PLE table served **live from the checkpoint safetensors shards**: a preadv thread-pool reader gathers each step's rows on the host in `prepare_inputs` into a fixed GPU staging buffer; the forward is views-only (cudagraph-safe). No assembled cache file is needed. |

Architecture credit: heavily inspired by Kai / 2Wild's [Qwen38-Flash-Next-4x3090]. I ported the staged design and preadv reader with PP support. Also see [vllm-ngram-pager] (mmap variant) and [iIIusi0n/qwen3.8-flash-next-cmp170hx] (native-AIO variant on the same GPU).

## Serving config

`configs/intel-1m-fp8-ssmbf16.yml` — the key lines:

```yaml
pipeline-parallel-size: 3
block-size: 1632             # % 12 == 0 (QSA ring); do not use 8192
kv-cache-dtype: fp8_e4m3
mamba-ssm-cache-dtype: bfloat16   # halves mamba pages; fp32 default eats the fp8 dividend
max-model-len: 1048576
hf-overrides:
  text_config:
    layer_types: [linear_attention ×3, qwen_sparse_attention] ×12   # transformers-5.18 bridge
    rope_parameters: {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 262144,
                      rope_theta: 10000000, partial_rotary_factor: 0.25,
                      mrope_interleaved: true, mrope_section: [11, 11, 10]}
```

Watch the boot log: `Padding mamba page size by ~2%` is healthy; a large or negative percentage means the `units-fix` patch isn't active.

## Validation

`tests/needle.py` runs the greedy needle suite used to qualify every change:

```
8k/64k/192k/240k/512k d0.5 ... PASS
1M (809k ptoks) d0.5 / d0.97 / d0.05 ... PASS   (exact codes, temperature=0)
17*23 → 391 (deterministic ×3)
```

## Performance

Measured cold-cache (unique random prompts so prefix caching can't fake
prefill), single stream, `temperature=0` — see
[`tests/bench-results-2026-10-04.md`](tests/bench-results-2026-10-04.md)
and rerun with [`tests/bench.py`](tests/bench.py):

| ctx | prefill tok/s | TTFT | decode tok/s |
|----:|--------------:|-----:|-------------:|
| 4k | 2,389 | 1.7 s | 73.2 |
| 16k | 3,210 | 5.1 s | 73.3 |
| 64k | 3,563 | 18.4 s | 74.9 |
| 256k | 3,345 | 78.4 s | 61.9 |
| 1M | **4,690** | 223 s | 59.8 |

Prefill per-token rate *improves* past ~64k — the QSA sparse-attention
crossover (top-2048 budget, no quadratic term; cf. the Qwen tech report's
7.6× prefill claim at 1M). Decode drops ~74→60 tok/s at 1M: the SM80
fp8-KV software-dequant tax.

## Troubleshooting

- **Boot OOM in `allocate_embedding_weight` (cudaHostAlloc)** → the staged env gate isn't active; check `VLLM_PLE_STAGED=1`.
- **`assert attn_page_size >= mamba_page_size`** → units-fix not mounted.
- **Needles pass at short range, fail ≥2k** → PLE delivery poisoned (wrong rows). With staged mode this shouldn't occur; if it does, `VLLM_PLE_ZERO=1` is the ablation knob that zeroes the table contribution (model still works — QSA attention carries retrieval).
- **TP2+ NCCL hang** → P2P; see hardware notes.
- **fp8 boot with block 3264** → units-fix cdiv half missing.

## Status / caveats

- MTP speculative decoding is **not** enabled (draft-layer KV would cost ~16% of the pool; pending fixes noted in the fork).
- The `interface.py` units fix and rank-ownership skip are candidates for upstreaming.
- Pin: image `nightly-0cbac6cd…`, fork `q38fn/base`. Newer nightlies have not been validated against this stack.

[Intel's INT4-weight AutoRound quant]: https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
[froggeric's extensively bugfixed Qwen3.8 chat templates]: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
[`clee/vllm-sm80`]: https://github.com/clee/vllm-sm80
[Qwen38-Flash-Next-4x3090]: https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090
[vllm-ngram-pager]: https://github.com/monochromegane/vllm-ngram-pager
[iIIusi0n/qwen3.8-flash-next-cmp170hx]: https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx
