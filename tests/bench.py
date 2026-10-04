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

CTXS = [1024, 4096, 16384, 65536, 262144, 1048576]
GEN = 64


def filler(n_tokens: int, seed: int) -> str:
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
    print(f"{'ctx':>9} {'prefill tok/s':>13} {'TTFT s':>8} {'decode tok/s':>12}")
    rows = []
    for i, ctx in enumerate(CTXS):
        prompt = filler(ctx, seed=1000 + i)  # unique per row
        ttft, n, dtail = stream_run(prompt, GEN)
        p_rate = ctx / ttft if ttft else float("nan")
        d_rate = (n - 1) / dtail if dtail > 0 and n > 1 else float("nan")
        print(f"{ctx:>9} {p_rate:>13.0f} {ttft:>8.2f} {d_rate:>12.1f}", flush=True)
        rows.append({"ctx": ctx, "prefill_tps": round(p_rate), "ttft_s": round(ttft, 2),
                     "decode_tps": round(d_rate, 1)})
    # short-prompt decode-only (no queue effect), 3 runs
    dec = []
    for j in range(3):
        ttft, n, dtail = stream_run(filler(512, seed=7000 + j), 256)
        if dtail > 0 and n > 1:
            dec.append((n - 1) / dtail)
    print(f"\nshort-ctx decode runs: {[round(x, 1) for x in dec]}")
    json.dump(rows, open("/tmp/bench-rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
