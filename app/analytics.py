"""Run analytics computed from agent_runs (PRD FR-6). Pure functions over rows; no HTTP.

Cost is reported only when the caller supplies a price. No price is assumed.
"""
import math
from collections import Counter


def _percentile(values, p):
    """Nearest-rank percentile: the value at rank ceil(p * n), 1-based."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[rank - 1]


def summarise(runs, price_in_per_1k=None, price_out_per_1k=None):
    """runs: list of dicts with status, total_latency_ms, input_tokens, output_tokens."""
    total = len(runs)
    statuses = Counter(r["status"] for r in runs)
    errors = statuses.get("llm_error", 0) + statuses.get("tool_error", 0)
    latencies = [r["total_latency_ms"] for r in runs
                 if r["total_latency_ms"] is not None and r["status"] != "rejected"]
    tokens_in = sum(r["input_tokens"] or 0 for r in runs)
    tokens_out = sum(r["output_tokens"] or 0 for r in runs)
    missing_usage = sum(1 for r in runs
                        if r["input_tokens"] is None and r["status"] not in ("rejected",))

    cost = None
    if price_in_per_1k is not None and price_out_per_1k is not None:
        cost = round(tokens_in / 1000 * price_in_per_1k
                     + tokens_out / 1000 * price_out_per_1k, 6)

    return {
        "runs": total,
        "by_status": dict(statuses),
        "error_rate": round(errors / total, 4) if total else None,
        "latency_ms": {
            "n": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "tokens": {"input": tokens_in, "output": tokens_out},
        "runs_without_usage": missing_usage,
        "estimated_cost_usd": cost,
    }
