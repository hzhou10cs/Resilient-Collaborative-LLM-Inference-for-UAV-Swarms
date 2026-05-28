"""Post-processing helpers for AeroKV traces."""

from __future__ import annotations

from collections.abc import Iterable

from .traces import TokenTraceRow, UAVTraceRow


def final_token_uav_rows(rows: Iterable[UAVTraceRow]) -> list[UAVTraceRow]:
    rows = list(rows)
    if not rows:
        return []
    final_token = max(r.token for r in rows)
    return [r for r in rows if r.token == final_token]


def mean_residual_energy_j(rows: Iterable[UAVTraceRow]) -> float:
    values = [r.energy_j for r in rows if r.uav_status != "failed"]
    return sum(values) / len(values) if values else 0.0


def system_expected_remaining_tokens(rows: Iterable[UAVTraceRow]) -> float:
    """Compute system expected remaining tokens from one token's UAV rows.

    The system value is the minimum per-UAV capacity among non-failed UAVs.
    This matches the value written in ``TokenTraceRow.system_expected_remaining_tokens``.
    """

    values = [r.expected_remaining_tokens for r in rows if r.uav_status != "failed"]
    return min(values) if values else 0.0


def cumulative_completion_time_s(rows: Iterable[TokenTraceRow]) -> float:
    rows = list(rows)
    return max((r.time_s for r in rows), default=0.0)
