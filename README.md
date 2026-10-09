# Qwen3.8-Flash-Next on 3× CMP170HX (SM80) with vLLM

A working recipe for serving [Intel's INT4-weight AutoRound quant] (Qwen4Exp, 180B-A6B MoE) on **3× NVIDIA CMP170HX (SM80, unlocked to 64 GiB each)** with:


| Capability | Status |
|---|---|
| W4A16 (AutoRound/INC) experts | ✅ GPTQ-Marlin, dequant verified bit-exact |
| 95.4 GiB BF16 PLE n-gram table | ✅ streamed from NVMe (never fully loaded to RAM/VRAM) |
| FP8 e4m3 KV cache | ✅ (software e4m3 on SM80) |
| Mamba SSM state | ✅ bf16 (`--mamba-ssm-cache-dtype bfloat16`) |
| Context length | ✅ 1,048,576 tokens via YaRN ×4 |
| KV pool | **7.36M tokens** (7.01× concurrent 1M sessions) |
| Fixed chat templates | [froggeric's extensively bugfixed Qwen3.8 chat templates] | 
| Decode | ~74 tok/s short-context, ~73 tok/s at 1M |
| Retrieval QA | exact needle recall at all depths through 1M |

The vLLM changes live in the fork [`clee/vllm-sm80`] (see [Patches](#patches)). This repo holds the walkthrough, launcher, serving config, and validation harness.

## Hardware notes (my specific machine)

- **No GPU P2P.** For me, CMP170HX pairs on different root buses never complete real peer transfers (BAR1 maps are created, DMA hangs). Always `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`; PP stages through host RAM via NCCL SHM, which works fine since there's very little data that actually travels between cards.
- **Pinned host memory ceiling ≈ 64 GiB.** My Threadripper machine only has 128GB of RAM, so the PLE table (95.4 GiB) can't really be pinned — that's why the NVMe path below exists.
- **No FP8 Triton pointers on SM80.** FP8 KV cache is stored as `uint8` and software-decoded (see `q38fn/fp8-kv-sm80`).

## Quick start

```bash
# 1. checkpoint on a local NVMe (needs ~169 GiB for the checkpoint, plus up to
#    max_bytes = 128 GiB for the NVMe KV tier — see *KV-cache offload* below)
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir /home/models/qwen38fn-w4a16-autoround

# 2. pull the pinned nightly the patches are based on
docker pull vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe

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
VLLM_PP_LAYER_PARTITION=16,16,16        # PLE layer (idx 1) stays on rank 0
VLLM_PLE_STAGED=1                       # NVMe PLE (shard-direct preadv)
QWEN4EXP_PLE_MODEL_DIR=/models/qwen38fn-w4a16-autoround
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1         # 1M > checkpoint's 262144 native
```

## Patches

Rebased onto the pinned upstream nightly (`7d0b4e57…` — see [Status]/[caveats]). Tip is `q38fn/stack`:

| Bookmark | What it does |
|---|---|
| `q38fn/units-fix` | Fixes a 3-instance units bug in `_align_hybrid_block_size`: one attention layer's page was compared against the all-layers mamba aggregate — spurious block doubling under fp8 KV, assert failure, and negative mamba padding. |
| `q38fn/fp8-kv-sm80` | fp8 e4m3 KV on SM80: uint8 cache storage + software decode in the QSA gather kernel, bit-level RNE encoder for writes. |
| `q38fn/ple-pipeline-parallel` | PLE checkpoints under PP>1: gate removal, per-worker `set_current_vllm_config` at load, rank-unowned tensor skip in `allocate_kv_cache`, MTP draft on last rank. |
| `q38fn/ple-nvme-staged` | The 95.4 GiB PLE table served **live from the checkpoint safetensors shards**: a preadv thread-pool reader gathers each step's rows on the host in `prepare_inputs` into a fixed GPU staging buffer; the forward is views-only (cudagraph-safe). No assembled cache file is needed. |
| `pr54327-fs-cap` | Bounds the fs KV tier with `max_bytes` + LRU eviction (`configs/…yml` sets 128 GiB); backport of upstream #54327. |
| (`q38fn/stack`) | Tip of the working stack. Worker-rank shared-memory slot selection across PP ranks (backport of upstream #58497). |

Two exploratory branches intentionally sit **off** the mainline: `q38fn/marlin-inv`
(batch-invariant Marlin MoE + GDN) and `kv-fp8-prom-probe` (e4m3 KV-magnitude probe).
They layer onto `q38fn/stack` and are re-pinnable with one `jj rebase`.

Architecture credit: heavily inspired by Kai / 2Wild's [Qwen38-Flash-Next-4x3090]. I ported the staged design and preadv reader with PP support. Also see [vllm-ngram-pager] (mmap variant) and [iIIusi0n/qwen3.8-flash-next-cmp170hx] (native-AIO variant on the same GPU).

## Serving config

`configs/intel-1m-fp8-ssmbf16.yml` — the key lines:

```yaml
pipeline-parallel-size: 3
block-size: 1728             # % 12 == 0 (QSA ring) && % 16 == 0 (mamba chunk grid)
kv-cache-dtype: fp8_e4m3
mamba-ssm-cache-dtype: bfloat16   # halves mamba pages; fp32 default eats the fp8 dividend
max-model-len: 1048576
hf-overrides:
  text_config:
    layer_types: # 12× [linear_attention ×3, qwen_sparse_attention]
    rope_parameters: {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 262144,
                      rope_theta: 10000000, partial_rotary_factor: 0.25,
                      mrope_interleaved: true, mrope_section: [11, 11, 10]}

# Tiered KV prefix cache: evicted GPU pages cascade down GPU -> CPU -> NVMe.
kv-transfer-config:
  kv_connector: OffloadingConnector
  kv_role: kv_both
  kv_connector_extra_config:
    spec_name: TieringOffloadingSpec
    cpu_bytes_to_use: 8589934592     # 8 GiB CPU LRU gateway (uses 8GiB in /dev/shm!)
    eviction_policy: lru
    secondary_tiers:
      - type: fs
        root_dir: /kv-disk           # host: /home/models/kv-disk
        max_bytes: 137438953472      # 128 GiB NVMe cap, LRU-evicted
```

I've pulled in a PR that isn't merged yet to enforce LRU-based max KV cache data usage on the NVMe. Blocks on disk are content-addressed, so **the cache survives restarts**, and, because the key is `sha256` of the serving geometry (`tp_size`/`pp_size`/`dtype`/block size), a config change orphans the old directory entirely: the tier will never open or prune it again. Those orphaned namespaces are the only thing `max_bytes` doesn't bound; deleting them is always safe.

Watch the boot log: `Padding mamba page size by ~8%` is healthy at block 1728; a far larger or negative percentage means the `units-fix` patch isn't active.

## Validation

`tests/needle.py` runs the greedy needle suite used to qualify every change:

```
8k/64k/192k/240k/512k d0.5 ... PASS
1M (809k ptoks) d0.5 / d0.97 / d0.05 ... PASS   (exact codes, temperature=0)
17*23 → 391 (deterministic ×3)
```

## Performance

Measured cold-cache (unique random prompts so prefix caching can't fake prefill), single stream, `temperature=0`, and the prompt's *actual* token count is confirmed against `/tokenize` (see [`tests/bench-results-2026-10-10.md`](tests/bench-results-2026-10-10.md); rerun with [`tests/bench.py`](tests/bench.py)):

| prompt tokens | prefill tok/s | TTFT | decode tok/s |
|---:|---:|---:|---:|
| 1,026 | 2,592 | 0.40 s | 74.4 |
| 4,084 | 3,388 | 1.21 s | 74.7 |
| 16,393 | 4,278 | 3.83 s | 76.6 |
| 64,845 | 4,831 | 13.42 s | 75.5 |
| 258,992 | **4,947** | **52.35 s** | 75.0 |
| 1,036,890 | **4,748** | 218 s | 72.9 |

Prefill per-token rate *improves* past ~64k — the QSA sparse-attention crossover (top-2048 budget, no quadratic term; cf. the Qwen tech report's 7.6× prefill claim at 1M). 1M decode holds at ~73 t/s; the old ~60-tok/s dip (the fp8-KV software-dequant tax) is gone on this stack.

## Troubleshooting

- **Boot OOM in `allocate_embedding_weight` (cudaHostAlloc)** → the staged env gate isn't active; check `VLLM_PLE_STAGED=1`.
- **`assert attn_page_size >= mamba_page_size`** → units-fix not mounted.
- **Needles pass at short range, fail ≥2k** → PLE delivery poisoned (wrong rows). With staged mode this shouldn't occur; if it does, `VLLM_PLE_ZERO=1` is the ablation knob that zeroes the table contribution (model still works — QSA attention carries retrieval).
- **TP2+ NCCL hang** → P2P; see hardware notes.
- **fp8 boot with block 3264** → units-fix cdiv half missing.
- **`/kv-disk` grows beyond `max_bytes` / `ENOSPC` in the offload log** → the active namespace prunes itself (LRU + a startup sweep); a *stale* namespace from an earlier serving geometry is never reclaimed by the tier — delete it manually (pure cache, always safe). See *KV-cache offload*.

## Status / caveats

- MTP speculative decoding is **not** enabled (draft-layer KV would cost ~16% of the pool; pending fixes noted in the fork).
- The `interface.py` units fix and rank-ownership skip are candidates for upstreaming.
- Pin: image `nightly-7d0b4e57…`, fork `q38fn/stack`. Newer nightlies have not been validated against this stack.

[Intel's INT4-weight AutoRound quant]: https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
[froggeric's extensively bugfixed Qwen3.8 chat templates]: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
[`clee/vllm-sm80`]: https://github.com/clee/vllm-sm80
[Qwen38-Flash-Next-4x3090]: https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090
[vllm-ngram-pager]: https://github.com/monochromegane/vllm-ngram-pager
[iIIusi0n/qwen3.8-flash-next-cmp170hx]: https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx
