#!/usr/bin/env python3
"""Skateboard benchmark (risorseartificiali.com/skateboard) via LiteLLM, no harness.

Sends the verbatim prompt (typos included) to an OpenAI-compatible
chat/completions endpoint and saves the resulting SVG.

Usage:
    python3 skateboard_bench.py [--prompt minimal|constrained] [--no-think]
        [--base URL] [--model NAME] [--max-tokens N]

Env: LITELLM_KEY / LITELLM_API_KEY (only needed when hitting the gateway).
"""
import argparse
import datetime as dt
import json
import os
import re
import socket
import sys
import urllib.request

BASE = os.environ.get("SKATEBOARD_BASE", "http://192.168.11.36:8889/v1").rstrip("/")
KEY = os.environ.get("LITELLM_KEY") or os.environ.get("LITELLM_API_KEY")
DEFAULT_MODEL = "glm-5.3-flash"

PROMPT_MINIMAL = (
    "generate an animated svg of a man doing skatevoard tricks on a pipe. "
    "Be mindful of the real phisics"
)
PROMPT_CONSTRAINED = """Generate a single self-contained animated SVG (SMIL or CSS animations, no JavaScript, no external resources) of a man doing skateboard tricks on a pipe (half-pipe / vert ramp). Be mindful of real physics.

The scene must include:
- A half-pipe / vert ramp with two quarter pipes facing each other
- A skater figure with a recognizable skateboard (deck + 4 trucks/wheels)
- The skater performing at least one aerial trick above the pipe's coping

The animation must respect real physics:
- Pendulum motion: the skater swings back and forth in the transitions
- Speed varies with height (slower at the top of the ramp, faster at the bottom)
- Gravity: airborne tricks follow a parabolic arc and land back on the ramp
- The board rotates during the trick (e.g. kickflip or 360 spin) and aligns again before landing

Constraints:
- Single <svg> code block only, no explanation
- SMIL or CSS animations only, no JavaScript, no external resources
- Loop must be seamless (the skater ends where the loop restarts)
- Include XML comments describing what each group represents"""

PROMPTS = {"minimal": PROMPT_MINIMAL, "constrained": PROMPT_CONSTRAINED}
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# TCP keepalive on every outgoing socket (120s idle, 60s interval, 10 probes).
# Long gateway streams get RST'ed at ~800-950s by an established-flow timer on
# the WAN path (BGP ECMP / conntrack); keepalive probes pin the flow.
import http.client

_orig_connect = http.client.HTTPConnection.connect


def _keepalive_connect(self):
    _orig_connect(self)
    s = self.sock
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 120)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)


http.client.HTTPConnection.connect = _keepalive_connect
if hasattr(http.client, "HTTPSConnection"):
    http.client.HTTPSConnection.connect = _keepalive_connect


def call_llm(base: str, model: str, prompt: str, max_tokens: int, attempts: int = 3,
             extra: dict | None = None) -> dict:
    """Single streaming chat/completions call (SSE), assembled into a response dict.

    Streaming keeps bytes flowing so long reasoning phases don't hit idle
    read timeouts on the gateway/proxy path.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if extra:
        payload.update(extra)
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                base + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            )
            content_parts, reasoning_parts = [], []
            finish, usage = None, {}
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices", []):
                        delta = ch.get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        # LiteLLM uses reasoning_content; raw vLLM uses reasoning
                        for rk in ("reasoning_content", "reasoning"):
                            if delta.get(rk):
                                reasoning_parts.append(delta[rk])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
            return {
                "choices": [{
                    "message": {"content": "".join(content_parts),
                                "reasoning_content": "".join(reasoning_parts)},
                    "finish_reason": finish,
                }],
                "usage": usage,
            }
        except Exception as e:  # noqa: BLE001 - retry any transport failure
            last_err = e
            print(f"  attempt {attempt}/{attempts} failed: {e}", file=sys.stderr, flush=True)
    raise RuntimeError(f"all {attempts} attempts failed, last error: {last_err}")


def extract_svg(text: str) -> str | None:
    """Pull the SVG out of the reply: strip code fences, take <svg>...</svg>."""
    if not text:
        return None
    m = re.search(r"<svg[\s\S]*?</svg>\s*", text, re.IGNORECASE)
    return m.group(0).strip() if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base", default=BASE, help="OpenAI-compatible base URL")
    ap.add_argument("--prompt", choices=PROMPTS, default="minimal")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--no-think", action="store_true",
                    help="disable GLM thinking via chat_template_kwargs")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    os.makedirs(OUTDIR, exist_ok=True)
    model_slug = args.model.split("/")[-1].replace(".", "-")
    if args.no_think:
        model_slug += "-nothink"
    out_svg = os.path.join(OUTDIR, f"{model_slug}-{args.prompt}.svg")
    out_meta = os.path.join(OUTDIR, f"{model_slug}-{args.prompt}.json")

    print(f"model={args.model} prompt={args.prompt} -> {out_svg}", flush=True)
    resp = call_llm(base, args.model, PROMPTS[args.prompt], args.max_tokens,
                    extra={"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None)

    choice = resp["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    reasoning = choice.get("message", {}).get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    usage = resp.get("usage", {})

    svg = extract_svg(content)
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "finish_reason": finish,
        "usage": usage,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "svg_len": len(svg) if svg else 0,
        "has_animation": bool(svg and re.search(r"<animate|animateTransform|animateMotion|@keyframes", svg)),
        "error": None if svg else "no SVG in response",
    }

    if svg:
        with open(out_svg, "w") as f:
            f.write(svg + "\n")
    if reasoning:
        meta["reasoning_head"] = reasoning[:300]
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    if not svg:
        print("ERROR: no SVG extracted; raw content saved for inspection", file=sys.stderr)
        with open(out_svg + ".raw.txt", "w") as f:
            f.write(content)
        return 1
    print(f"OK: {out_svg} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
