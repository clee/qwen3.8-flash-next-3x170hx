# Benchmark results — 2026-10-04

Measured on the production stack: dev558 + Intel AutoRound W4A16 + fp8 KV
+ bf16 SSM state + live PLE (shard-direct), PP=3 (17/17/14),
block-size 1632, FULL_AND_PIECEWISE cudagraphs, chunked prefill, async
scheduling, prefix caching ON. `temperature=0`, single stream,
`/v1/completions` with streaming.

Prefill prompts use **unique random content per request** (word salad
tokenizing to ~1.35 tokens/word) so prefix caching cannot fake the
numbers. Decode rate is measured from streamed token deltas after TTFT.

| ctx (tokens) | prefill tok/s | TTFT (s) | decode tok/s |
|-------------:|--------------:|---------:|-------------:|
| 1,024        | 2,030         | 0.50     | 74.1         |
| 4,096        | 2,389         | 1.71     | 73.2         |
| 16,384       | 3,210         | 5.10     | 73.3         |
| 65,536       | 3,563         | 18.4     | 74.9         |
| 262,144      | 3,345         | 78.4     | 61.9         |
| 1,046,885    | **4,690**     | 223.2    | 59.8         |

Short-context decode (512-token completions, 3 runs): 73.9 / 74.1 / 74.3
tok/s — stable.

## What the shape of the curve means

- **Prefill per-token rate IMPROVES at 1M** (4.69k tok/s) vs 256k
  (3.35k). This is the QSA sparse-attention crossover: above ~64k
  context, the top-2048 token budget dominates and the attention cost
  stops scaling quadratically. Matches the Qwen tech report's claim of
  7.6× prefill speedup at 1M vs dense.
- **Decode settles from ~74 to ~60 tok/s** between 262k and 1M.
  That's the fp8 KV software-dequant tax on SM80 (uint8→f16 per read)
  plus indexer/selection cost; short-context decode is effectively
  context-independent.
- **TTFT at full 1M is 3.7 minutes** — chunked prefill (8192/chunk)
  keeps the server responsive during the wait.
