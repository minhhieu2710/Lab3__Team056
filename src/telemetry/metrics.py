import time
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.telemetry.logger import logger


# ------------------------------------------------------------------
# Pricing Table (USD per 1K tokens)  —  Updated June 2025
# ------------------------------------------------------------------
# Sources:
#   OpenAI  – https://openai.com/pricing
#   Google  – https://ai.google.dev/pricing
#   Local   – Free (electricity only, estimated at ~$0.001/1K tokens)

PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {
        "prompt": 0.0025,       # $2.50 / 1M input tokens
        "completion": 0.0100,   # $10.00 / 1M output tokens
    },
    "gpt-4o-mini": {
        "prompt": 0.00015,      # $0.15 / 1M input tokens
        "completion": 0.0006,   # $0.60 / 1M output tokens
    },
    "gpt-4-turbo": {
        "prompt": 0.01,         # $10.00 / 1M input tokens
        "completion": 0.03,     # $30.00 / 1M output tokens
    },
    "gpt-3.5-turbo": {
        "prompt": 0.0005,
        "completion": 0.0015,
    },

    # Google Gemini models
    "gemini-1.5-flash": {
        "prompt": 0.000075,     # $0.075 / 1M input tokens
        "completion": 0.0003,   # $0.30 / 1M output tokens
    },
    "gemini-1.5-pro": {
        "prompt": 0.00125,      # $1.25 / 1M input tokens
        "completion": 0.005,    # $5.00 / 1M output tokens
    },
    "gemini-2.0-flash": {
        "prompt": 0.0001,
        "completion": 0.0004,
    },

    # Local models (estimated electricity cost)
    "local": {
        "prompt": 0.0,
        "completion": 0.0,
    },
}

# Fallback when the model is not in the pricing table
_DEFAULT_PRICING = {"prompt": 0.001, "completion": 0.002}


class PerformanceTracker:
    """
    Industry-standard performance tracker for LLM-powered agents.

    Tracks per-request metrics (tokens, latency, cost) and provides
    aggregate statistics (P50/P99 latency, total cost, token efficiency)
    for comparison between agent versions or providers.
    """

    def __init__(self):
        self.session_metrics: List[Dict[str, Any]] = []
        self.session_start: datetime = datetime.utcnow()

    # ------------------------------------------------------------------
    # Per-request tracking
    # ------------------------------------------------------------------

    def track_request(
        self,
        provider: str,
        model: str,
        usage: Dict[str, int],
        latency_ms: int,
    ) -> Dict[str, Any]:
        """
        Log a single LLM request and its telemetry.

        Args:
            provider:   'openai', 'google', or 'local'
            model:      Model identifier (e.g. 'gpt-4o')
            usage:      Dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
            latency_ms: Round-trip time in milliseconds

        Returns:
            The metric dict that was recorded.
        """
        cost = self.calculate_cost(model, usage)

        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate_usd": round(cost, 8),
            "token_efficiency": self._token_efficiency(usage),
        }

        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)
        return metric

    # ------------------------------------------------------------------
    # Cost Calculation
    # ------------------------------------------------------------------

    def calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate the estimated cost (USD) for a single LLM request
        based on the model's pricing and token usage.

        Uses the PRICING lookup table. Falls back to a conservative
        default if the model name is unknown.
        """
        # Try exact match first, then check if model name contains a known key
        pricing = PRICING.get(model)
        if pricing is None:
            for key in PRICING:
                if key in model.lower():
                    pricing = PRICING[key]
                    break

        if pricing is None:
            pricing = _DEFAULT_PRICING

        prompt_cost = (usage.get("prompt_tokens", 0) / 1000) * pricing["prompt"]
        completion_cost = (usage.get("completion_tokens", 0) / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    # ------------------------------------------------------------------
    # Token Efficiency
    # ------------------------------------------------------------------

    @staticmethod
    def _token_efficiency(usage: Dict[str, int]) -> float:
        """
        Ratio of completion tokens to total tokens.

        A higher ratio means the model is generating more "useful" output
        relative to the input. Low ratios may indicate overly verbose
        system prompts or unnecessary context.

        Returns 0.0 when total_tokens is 0.
        """
        total = usage.get("total_tokens", 0)
        if total == 0:
            return 0.0
        return round(usage.get("completion_tokens", 0) / total, 4)

    # ------------------------------------------------------------------
    # Aggregate Session Statistics
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return an aggregate summary of all tracked requests in this session.

        Includes:
        - Total & average tokens
        - Latency percentiles (P50, P95, P99)
        - Total estimated cost
        - Average token efficiency
        - Request count & session duration
        """
        if not self.session_metrics:
            return {"status": "no_data", "request_count": 0}

        latencies = [m["latency_ms"] for m in self.session_metrics]
        total_tokens_list = [m["total_tokens"] for m in self.session_metrics]
        costs = [m["cost_estimate_usd"] for m in self.session_metrics]
        efficiencies = [m["token_efficiency"] for m in self.session_metrics]

        # Percentile calculation
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        summary = {
            "session_duration_s": (datetime.utcnow() - self.session_start).total_seconds(),
            "request_count": n,
            # --- Tokens ---
            "total_tokens": sum(total_tokens_list),
            "avg_tokens_per_request": round(statistics.mean(total_tokens_list), 1),
            "total_prompt_tokens": sum(m["prompt_tokens"] for m in self.session_metrics),
            "total_completion_tokens": sum(m["completion_tokens"] for m in self.session_metrics),
            # --- Latency ---
            "latency_p50_ms": latencies_sorted[n // 2],
            "latency_p95_ms": latencies_sorted[int(n * 0.95)] if n >= 2 else latencies_sorted[-1],
            "latency_p99_ms": latencies_sorted[int(n * 0.99)] if n >= 2 else latencies_sorted[-1],
            "latency_avg_ms": round(statistics.mean(latencies), 1),
            "latency_max_ms": max(latencies),
            # --- Cost ---
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_request_usd": round(statistics.mean(costs), 8),
            # --- Efficiency ---
            "avg_token_efficiency": round(statistics.mean(efficiencies), 4),
        }

        logger.log_event("SESSION_SUMMARY", summary)
        return summary

    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """
        Group metrics by (provider, model) and return a list of summaries.
        Useful for building the Chatbot-vs-Agent or Provider-vs-Provider
        comparison table in the group report.
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in self.session_metrics:
            key = f"{m['provider']}:{m['model']}"
            groups.setdefault(key, []).append(m)

        table = []
        for key, metrics in groups.items():
            provider, model = key.split(":", 1)
            latencies = [m["latency_ms"] for m in metrics]
            table.append({
                "provider": provider,
                "model": model,
                "requests": len(metrics),
                "total_tokens": sum(m["total_tokens"] for m in metrics),
                "avg_latency_ms": round(statistics.mean(latencies), 1),
                "total_cost_usd": round(sum(m["cost_estimate_usd"] for m in metrics), 6),
                "avg_efficiency": round(
                    statistics.mean(m["token_efficiency"] for m in metrics), 4
                ),
            })

        return table

    def reset(self):
        """Clear all metrics for a new session (e.g., Agent v1 → v2)."""
        self.session_metrics = []
        self.session_start = datetime.utcnow()
        logger.log_event("METRICS_RESET", {"timestamp": self.session_start.isoformat()})


# ------------------------------------------------------------------
# Global tracker instance (shared across the application)
# ------------------------------------------------------------------
tracker = PerformanceTracker()
