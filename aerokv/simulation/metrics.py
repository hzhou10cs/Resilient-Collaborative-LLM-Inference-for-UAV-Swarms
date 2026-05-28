"""Post-processing helpers for AeroKV traces."""

from __future__ import annotations

from collections.abc import Iterable

from .traces import TokenTraceRow, UAVTraceRow


def mean_residual_energy_j(rows: Iterable[UAVTraceRow]) -> float:
    values = [r.energy_j for r in rows if r.uav_status != "failed"]
    return sum(values) / len(values) if values else 0.0


def expected_remaining_token_capacity(rows: Iterable[UAVTraceRow], pipeline_latency_s: float) -> float:
    if pipeline_latency_s <= 0:
        return 0.0
    caps = []
    for r in rows:
        if r.uav_status == "failed":
            continue
        per_token = r.flight_energy_j + r.compute_energy_j + r.tx_energy_j
        if per_token > 0:
            caps.append(r.energy_j / per_token)
    return min(caps) if caps else 0.0


def cumulative_completion_time_s(rows: Iterable[TokenTraceRow]) -> float:
    rows = list(rows)
    return max((r.time_s for r in rows), default=0.0)
