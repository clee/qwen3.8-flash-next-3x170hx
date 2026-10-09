#!/usr/bin/env python3
"""Single-stream prefill/decode benchmarks for the SM80 Qwen4Exp stack.

Uses unique random content per request so prefix caching can't fake prefill
numbers. Decode rate is measured from streamed token deltas (TTFT excluded).
"""

import json
import random
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu one two three four five six seven eight nine ten").split()

# Nominal targets; the measured token count for each is what we actually report.
# Word-salad token density ≈ 1.344 tokens per nominal unit (measured via /tokenize),
# and every corpus is re-measured per run so neither the fixed-seed cache nor the
# persistent offload tier can serve a stale run's content.
CTXS = [1024, 4096, 16384, 65536, 262144, 1048576 - 64]   # last row: fit under max_model_len with GEN=64 slack
GEN = 64


def meas_tokens(text):
    body = json.dumps({"model": "qwen3.8", "prompt": text}).encode()
    req = urllib.request.Request(f"{BASE}/tokenize", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["count"]


def fit_prompt(target_tokens, seed):
    """Generate a word-salad corpus, confirm its token count via /tokenize,
    and clip toward the ceiling on the actual measured count (never trust the
    char-heuristic `n*5` guess against max_model_len)."""
    def filler(n_tokens):
        rng = random.Random(seed)
        out = []
        approx = 0
        while approx < n_tokens * 5:
            w = rng.choice(WORDS)
            out.append(w)
            approx += len(w) + 1
            if len(out) % 50 == 0:
                out.append(str(rng.randint(0, 10**9)))
        return " ".join(out)

    # ratio word-salad ≈ 1.344 t/nominal; start a hair under target on that basis
    nominal = target_tokens if target_tokens < 65536 else max(1, int(target_tokens / 1.36))
    p = filler(nominal)
    ptoks = meas_tokens(p)
    # one correction pass if we're over target by more than we want
    if ptoks > target_tokens:
        nominal = max(1, int(nominal * target_tokens / ptoks))
        p = filler(nominal)
        ptoks = meas_tokens(p)
    return p, ptoks


def stream_run(prompt: str, max_tokens: int):
    body = {"model": "qwen3.8", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True}
    req = urllib.request.Request(f"{BASE}/v1/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_tok = 0
    last = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if not d.get("choices"):
                continue
            tok = d["choices"][0].get("text", "")
            if tok:
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                last = now
                n_tok += 1
    return ttft, n_tok, (last - t0 - ttft) if last and ttft else 0


def main():
    run_seed = random.Random().randint(1, 10**9)
    print(f"run_seed={run_seed}")
    print(f"{'ctx (actual)':>13} {'prefill tok/s':>13} {'TTFT s':>8} {'decode tok/s':>12}")
    rows = []
    for i, ctx in enumerate(CTXS):
        prompt, ptoks = fit_prompt(ctx, seed=run_seed + i)  # unique per row and per run
        ttft, n, dtail = stream_run(prompt, GEN)
        p_rate = ptoks / ttft if ttft else float("nan")
        d_rate = (n - 1) / dtail if dtail > 0 and n > 1 else float("nan")
        print(f"{ptoks:>13} {p_rate:>13.0f} {ttft:>8.2f} {d_rate:>12.1f}", flush=True)
        rows.append({"ctx_target": ctx, "ctx_actual": ptoks,
                     "prefill_tps": round(p_rate), "ttft_s": round(ttft, 2),
                     "decode_tps": round(d_rate, 1)})
    # short-prompt decode-only (no queue effect), 3 runs
    dec = []
    for j in range(3):
        prompt, _ = fit_prompt(512, seed=run_seed + 7000 + j)
        ttft, n, dtail = stream_run(prompt, 256)
        if dtail > 0 and n > 1:
            dec.append((n - 1) / dtail)
    print(f"\nshort-ctx decode runs: {[round(x, 1) for x in dec]}")
    json.dump(rows, open("/tmp/bench-rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
