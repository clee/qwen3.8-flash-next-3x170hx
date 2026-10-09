# Benchmark results — 2026-10-10

Measured on the live 3× stack: image `nightly-7d0b4e57…`, fork checkout
`q38fn/stack` tip (fs-tier capacity cap + worker-rank shm-slot backport,
upstream #54327/#58497), PP=3 (16/16/16), block-size 1728, `fp8_e4m3` KV,
bf16 SSM, and the tiered KV offload (8 GiB CPU LRU gateway → NVMe fs at
`/kv-disk` with `max_bytes: 128 GiB`). Single stream, `temperature=0`,
`/v1/completions` with streaming.

Harness fixes this run ([`tests/bench.py`](bench.py)): word-salad density
now *confirmed via `/tokenize`* (~1.35 tokens per nominal unit under this
tokenizer), and corpus fits the 1M row under `max_model_len` after measuring
(before, the char-heuristic guess overshot by 1 token and the server 400'd).
Seeds are fresh per run, so the persistent offload tier / prefix cache cannot
replay yesterday's content and inflate prefill.

| ctx (actual tokens) | prefill tok/s | TTFT (s) | decode tok/s |
|--------------------:|--------------:|---------:|-------------:|
| 1,026        | 2,592 | 0.40    | 74.4 |
| 4,084        | 3,388 | 1.21    | 74.7 |
| 16,393       | 4,278 | 3.83    | 76.6 |
| 64,845       | 4,831 | 13.42   | 75.5 |
| 258,992      | 4,947 | **52.35** | 75.0 |
| 1,036,890    | **4,748** | 218.38 | 72.9 |

Short-context decode (3 runs): 74.7 / 75.3 / 72.3 tok/s.

## Cross-run comparison (2026-10-04 → today)

| row | prefill tok/s | TTFT (s) | decode tok/s |
|-----|--------------:|---------:|-------------:|
| ~16k   | 3,210 → 4,278 | 5.10 → 3.83 | 73.3 → 76.6 |
| ~64k   | 3,563 → 4,831 | 18.4 → 13.42 | 74.9 → 75.5 |
| ~256k  | 3,345 → 4,947 | 78.4 → **52.35** | 61.9 → 75.0 |
| ~1M    | **4,690 → 4,748** | 223.2 → 218.38 | **59.8 → 72.9** |

Note on apples-to-apples: the 10-04 numbers were taken with the previous image
(`dev558`-era) and, per the old harness, used *nominal* `ctx` labels rather than
tokenizer-measured counts; today's `ctx` column is the actual token count the
server saw. Decode is the cleanest comparison because it is time-based either way.

## What the shape of the curve means today

- **Long-context prefill no longer degrades.** TTFT at 256k went 78.4 s → 52.35 s
  (~33% faster) while prefill rate rose 3,345 → 4,947 tok/s. The QSA sparse-
  attention crossover is still the headline. The 1M point now *continues the
  trend*: 4,748 tok/s, TTFT 218 s.
- **Long-context decode floor rose from ~60 to ~73 t/s.** Previously the SM80
  fp8-dequant tax dragged 1M decode to ~60; today it sits at 72.9, within ~2%
  of the short-context decode rate. The long-context decode penalty on this stack
  is effectively gone.
- Short-context decode is unchanged (~74 t/s), as expected: it's not offloaded
  and not sparse-limited.
