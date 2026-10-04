#!/usr/bin/env python3
"""Greedy needle suite for the Qwen3.8-Flash-Next long-context stack.

Validates exact mid-context retrieval at increasing total lengths and
needle depths. PASS = the exact code string appears in the content.
Usage: python3 needle.py [http://host:port]
"""

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank every "
    "morning. Local farmers report that the harvest season brings travelers "
    "from distant valleys. Merchants trade wool, grain, and copper tools at "
    "the weekly market by the old stone bridge. "
) * 120_000  # ~5.15 chars/token; enough for a 1M-token prompt

SUITE = [
    (8192, 0.5),
    (65_536, 0.5),
    (196_608, 0.5),
    (245_760, 0.5),
    (524_288, 0.5),
    (1_048_576, 0.5),
    (1_048_576, 0.97),
    (1_048_576, 0.05),
]


def ask(prompt: str, max_tokens: int = 60) -> str:
    body = {
        "model": "qwen3.8",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def run(tokens: int, depth: float, code: str) -> bool:
    body = FILLER[: tokens * 5]  # ~5.15 chars/token, trim to budget below
    needle = f"The access code for vault N is {code}. "
    pos = int(len(body) * depth)
    body = body[:pos] + needle + body[pos:]
    body += "\n\nWhat is the access code for vault N? Reply with just the code."
    t0 = time.time()
    content = ask(body)
    ok = code in content
    print(
        f"{'PASS' if ok else 'FAIL'}  {tokens:>9,} d{depth:<4} "
        f"({time.time() - t0:.0f}s)  {content[:60]!r}",
        flush=True,
    )
    return ok


def main() -> int:
    assert ask("What is 17*23? Answer with just the number.", 100).strip() == "391", (
        "sanity failed: 17*23"
    )
    print("PASS  sanity 17*23 = 391")
    results = [run(tok, depth, f"NEEDLE-{i:04d}-X") for i, (tok, depth) in enumerate(SUITE)]
    print(f"\n{sum(results)}/{len(results)} needles passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
