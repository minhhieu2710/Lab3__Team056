"""
benchmark_providers.py
======================
Phase 4 – Provider Switching & Latency Comparison

Compares OpenAI and Gemini on the same set of prompts.
Falls back to a SimulatedProvider (realistic mock) when
real API keys are not configured, so the script runs offline.

Usage:
    python benchmark_providers.py
    # With real keys set in .env:
    python benchmark_providers.py --real
"""

import os
import time
import json
import random
import argparse
from typing import Dict, Any, Optional, Generator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────
# Simulated Providers (runs without API keys)
# ─────────────────────────────────────────────

class SimulatedProvider:
    """
    Realistic simulation of an LLM provider.
    Models real-world p50/p95 latency and token usage.
    """
    PROFILES = {
        "gpt-4o": {
            "latency_mean_ms": 1800,
            "latency_std_ms":  400,
            "tokens_per_word": 1.3,
            "provider": "openai",
            "cost_prompt":      0.0025 / 1000,
            "cost_completion":  0.010  / 1000,
        },
        "gpt-4o-mini": {
            "latency_mean_ms": 900,
            "latency_std_ms":  200,
            "tokens_per_word": 1.3,
            "provider": "openai",
            "cost_prompt":      0.000150 / 1000,
            "cost_completion":  0.000600 / 1000,
        },
        "gemini-1.5-flash": {
            "latency_mean_ms": 700,
            "latency_std_ms":  150,
            "tokens_per_word": 1.2,
            "provider": "google",
            "cost_prompt":      0.000075 / 1000,
            "cost_completion":  0.000300 / 1000,
        },
        "gemini-1.5-pro": {
            "latency_mean_ms": 1400,
            "latency_std_ms":  300,
            "tokens_per_word": 1.2,
            "provider": "google",
            "cost_prompt":      0.00125 / 1000,
            "cost_completion":  0.005   / 1000,
        },
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name not in self.PROFILES:
            raise ValueError(f"Unknown simulated model: {model_name}")
        self._profile = self.PROFILES[model_name]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        p = self._profile
        latency_ms = max(200, int(random.gauss(p["latency_mean_ms"], p["latency_std_ms"])))

        prompt_tokens = int(len(prompt.split()) * p["tokens_per_word"])
        completion_tokens = int(random.gauss(80, 20))   # avg response ~80 tokens
        completion_tokens = max(20, completion_tokens)

        # Simulate a plausible reply
        answer = (
            f"[Simulated {self.model_name} response] "
            f"Based on the prompt, here is my answer: ... "
            f"(latency simulated at {latency_ms}ms)"
        )

        return {
            "content": answer,
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      prompt_tokens + completion_tokens,
            },
            "latency_ms": latency_ms,
            "provider":   p["provider"],
            "model":      self.model_name,
            "simulated":  True,
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        result = self.generate(prompt, system_prompt)
        for word in result["content"].split():
            time.sleep(0.01)
            yield word + " "


# ─────────────────────────────────────────────
# Factory: real or simulated provider
# ─────────────────────────────────────────────

def build_provider(model_name: str, use_real: bool = False):
    """Return a real or simulated provider based on --real flag & env vars."""
    if not use_real:
        return SimulatedProvider(model_name)

    provider_key = SimulatedProvider.PROFILES.get(model_name, {}).get("provider", "")
    try:
        if provider_key == "openai":
            from src.core.openai_provider import OpenAIProvider
            key = os.getenv("OPENAI_API_KEY", "")
            if not key or key == "your_openai_api_key_here":
                print(f"  [WARN] OPENAI_API_KEY not set — falling back to simulation for {model_name}")
                return SimulatedProvider(model_name)
            return OpenAIProvider(model_name=model_name, api_key=key)

        elif provider_key == "google":
            from src.core.gemini_provider import GeminiProvider
            key = os.getenv("GEMINI_API_KEY", "")
            if not key or key == "your_gemini_api_key_here":
                print(f"  [WARN] GEMINI_API_KEY not set — falling back to simulation for {model_name}")
                return SimulatedProvider(model_name)
            return GeminiProvider(model_name=model_name, api_key=key)

    except ImportError as e:
        print(f"  [WARN] Import error ({e}) — falling back to simulation for {model_name}")
        return SimulatedProvider(model_name)

    return SimulatedProvider(model_name)


# ─────────────────────────────────────────────
# Benchmark Logic
# ─────────────────────────────────────────────

TEST_PROMPTS = [
    {
        "id":   1,
        "desc": "Simple math",
        "text": "What is 25 * 4 + 10?",
    },
    {
        "id":   2,
        "desc": "Multi-step e-commerce reasoning",
        "text": (
            "A laptop costs $1200. I have a 15% discount coupon and "
            "the shipping fee is $20. What is the final total?"
        ),
    },
    {
        "id":   3,
        "desc": "Short creative writing",
        "text": "Write a 2-sentence description of an AI assistant for an e-commerce store.",
    },
]

MODELS_TO_COMPARE = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

PRICING = {m: SimulatedProvider.PROFILES[m] for m in MODELS_TO_COMPARE}


def calculate_cost(model: str, usage: Dict) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (
        usage["prompt_tokens"]     * p["cost_prompt"] +
        usage["completion_tokens"] * p["cost_completion"]
    )


def run_benchmark(use_real: bool = False, n_runs: int = 3):
    print("\n" + "=" * 65)
    print("  PHASE 4 — Provider Switching & Latency Benchmark")
    mode = "REAL API" if use_real else "SIMULATED (offline)"
    print(f"  Mode: {mode}  |  Runs per prompt: {n_runs}")
    print("=" * 65)

    # results[model] = list of dicts with latency, cost, tokens per prompt-run
    results: Dict[str, list] = {m: [] for m in MODELS_TO_COMPARE}

    for model_name in MODELS_TO_COMPARE:
        print(f"\n[Model: {model_name}]")
        provider = build_provider(model_name, use_real)
        
        for prompt in TEST_PROMPTS:
            for run in range(n_runs):
                try:
                    r = provider.generate(prompt["text"])
                    cost = calculate_cost(model_name, r["usage"])
                    results[model_name].append({
                        "prompt_id":         prompt["id"],
                        "prompt_desc":       prompt["desc"],
                        "run":               run + 1,
                        "latency_ms":        r["latency_ms"],
                        "prompt_tokens":     r["usage"]["prompt_tokens"],
                        "completion_tokens": r["usage"]["completion_tokens"],
                        "total_tokens":      r["usage"]["total_tokens"],
                        "cost_usd":          cost,
                        "simulated":         r.get("simulated", False),
                    })
                    print(f"  Prompt {prompt['id']} run {run+1}: "
                          f"{r['latency_ms']:>5}ms | "
                          f"{r['usage']['total_tokens']:>4} tok | "
                          f"${cost:.6f}")
                except Exception as e:
                    print(f"  Prompt {prompt['id']} run {run+1}: ERROR — {e}")

    return results


def print_summary(results: Dict[str, list]):
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE — Average across all prompts")
    print("=" * 65)
    print(f"{'Model':<22} {'Provider':<8} {'Lat(ms)':<10} {'Tokens':<8} {'Cost/req':<12} {'Cost/1K tok'}")
    print("-" * 65)

    summary_rows = []
    for model_name, runs in results.items():
        if not runs:
            continue
        n = len(runs)
        avg_lat   = sum(r["latency_ms"]    for r in runs) / n
        avg_tok   = sum(r["total_tokens"]  for r in runs) / n
        avg_cost  = sum(r["cost_usd"]      for r in runs) / n
        provider  = SimulatedProvider.PROFILES[model_name]["provider"]

        cost_per_1k = (avg_cost / avg_tok * 1000) if avg_tok > 0 else 0

        summary_rows.append({
            "model":        model_name,
            "provider":     provider,
            "avg_lat_ms":   avg_lat,
            "avg_tokens":   avg_tok,
            "avg_cost_usd": avg_cost,
            "cost_per_1k":  cost_per_1k,
        })
        print(f"{model_name:<22} {provider:<8} {avg_lat:>7.0f}ms "
              f"{avg_tok:>7.0f}   ${avg_cost:.6f}   ${cost_per_1k:.6f}")

    print("\n  KEY FINDINGS")
    print("  -----------")
    # Fastest
    fastest = min(summary_rows, key=lambda x: x["avg_lat_ms"])
    print(f"  - Fastest:    {fastest['model']} ({fastest['avg_lat_ms']:.0f}ms avg)")
    # Cheapest
    cheapest = min(summary_rows, key=lambda x: x["avg_cost_usd"])
    print(f"  - Cheapest:   {cheapest['model']} (${cheapest['avg_cost_usd']:.6f}/req)")
    # Best value (cost-per-1k-tokens)
    best_val = min(summary_rows, key=lambda x: x["cost_per_1k"])
    print(f"  - Best value: {best_val['model']} (${best_val['cost_per_1k']:.6f}/1K tok)")

    return summary_rows


def save_report(results: Dict[str, list], summary_rows: list):
    report_path = "report/Phase4_Provider_Comparison.json"
    os.makedirs("report", exist_ok=True)
    payload = {
        "phase": 4,
        "description": "Provider Switching & Latency Benchmark",
        "models_compared": MODELS_TO_COMPARE,
        "summary": summary_rows,
        "raw_results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Provider Benchmark")
    parser.add_argument("--real", action="store_true",
                        help="Use real API providers (requires valid API keys in .env)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per prompt per model (default: 3)")
    args = parser.parse_args()

    results = run_benchmark(use_real=args.real, n_runs=args.runs)
    summary = print_summary(results)
    save_report(results, summary)

    print("\n  DEFAULT_PROVIDER switching demo:")
    for pname in ["openai", "google"]:
        os.environ["DEFAULT_PROVIDER"] = pname
        model_map = {"openai": "gpt-4o-mini", "google": "gemini-1.5-flash"}
        m = model_map[pname]
        prov = build_provider(m, use_real=args.real)
        r = prov.generate("What is 2 + 2?")
        print(f"  DEFAULT_PROVIDER={pname:<7} -> {m:<22} latency={r['latency_ms']}ms")


if __name__ == "__main__":
    main()
