"""Smoke test for the local vLLM server (block B0).

Checks from the Mac that the vLLM server on the Windows PC:
  1. is reachable and reports its model,
  2. handles N parallel chat requests (prints tokens/s),
  3. returns a proper tool call (needed by the agent in B1).

Usage:
    export TIPPING_LLM_BASE_URL=http://<windows-pc-ip>:8000/v1
    export TIPPING_LLM_API_KEY=<the key you passed to vllm serve>
    uv run python scripts/vllm_smoke.py --n 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

BASE_URL = os.environ.get("TIPPING_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("TIPPING_LLM_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

PROMPTS = [
    "Explain in two sentences why agent evaluation needs repeated runs.",
    "Name three tools a shopping assistant agent might need.",
    "What is a confidence interval? One short paragraph.",
    "Write a haiku about a flaky test.",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def get_model(client: httpx.AsyncClient) -> str:
    model = os.environ.get("TIPPING_LLM_MODEL")
    if model:
        return model
    r = await client.get(f"{BASE_URL}/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


async def one_request(client: httpx.AsyncClient, model: str, i: int, max_tokens: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    r = await client.post(f"{BASE_URL}/chat/completions", json=body)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    return latency, r.json()["usage"]["completion_tokens"]


async def throughput(client: httpx.AsyncClient, model: str, n: int, max_tokens: int) -> None:
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_request(client, model, i, max_tokens) for i in range(n)))
    wall = time.perf_counter() - t0
    latencies = [lat for lat, _ in results]
    tokens = sum(tok for _, tok in results)
    print(f"[2/3] {n} parallel requests: {tokens} completion tokens in {wall:.1f}s")
    print(f"      throughput: {tokens / wall:.0f} tokens/s")
    print(f"      latency: median {statistics.median(latencies):.1f}s, max {max(latencies):.1f}s")


async def tool_call(client: httpx.AsyncClient, model: str) -> bool:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in Amsterdam right now?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0,
    }
    r = await client.post(f"{BASE_URL}/chat/completions", json=body)
    r.raise_for_status()
    calls = r.json()["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        print("[3/3] FAIL: no tool call returned. Check --enable-auto-tool-choice / parser.")
        return False
    fn = calls[0]["function"]
    args = json.loads(fn["arguments"])
    print(f"[3/3] tool call OK: {fn['name']}({args})")
    return fn["name"] == "get_weather" and "city" in args


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32, help="number of parallel requests")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    async with httpx.AsyncClient(headers=HEADERS, timeout=300) as client:
        try:
            model = await get_model(client)
        except httpx.HTTPError as e:
            print(f"[1/3] FAIL: cannot reach {BASE_URL}: {e}")
            return 1
        print(f"[1/3] server OK at {BASE_URL}, model: {model}")
        await throughput(client, model, args.n, args.max_tokens)
        ok = await tool_call(client, model)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
