"""Shared experiment helpers."""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from aerokv.baselines import no_protection_plan, overlap_only_plan, snapshot_only_plan
from aerokv.config import ExperimentConfig
from aerokv.experiments.scenarios import make_initial_layout, make_initial_ring, make_standard_system
from aerokv.optimizers.p1_provisioning import solve_p1_provisioning
from aerokv.simulation.engine import make_fixed_protection_plan
from aerokv.simulation.events import FailureEvent, generate_poisson_failure_events
from aerokv.specs import ProtectionPlan, SystemSpec


def build_standard_context(seed: int = 2026, config: ExperimentConfig | None = None):
    cfg = config if config is not None else ExperimentConfig(seed=seed)
    system = make_standard_system(seed=seed, config=cfg)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    return cfg, system, layout, ring


def print_uav_profiles(system: SystemSpec, config: ExperimentConfig | None = None) -> None:
    base_latency_s = None if config is None else config.per_layer_latency_min_ms / 1000.0
    for uav in system.uavs:
        if config is not None and config.uav_compute_latency_multipliers is not None:
            multiplier = config.uav_compute_latency_multipliers[uav.uav_id]
        elif base_latency_s is not None and base_latency_s > 0:
            multiplier = uav.per_layer_latency_s / base_latency_s
        else:
            multiplier = 1.0
        print(
            f"[UAV][profile] id={uav.uav_id} "
            f"compute_latency_multiplier={multiplier:.3f} "
            f"per_layer_latency_s={uav.per_layer_latency_s:.9f} "
            f"link_mbps={uav.link_bps / 1e6:.3f} "
            f"memory_gb={uav.memory_budget_bytes / (1024 ** 3):.3f} "
            f"energy_j={uav.initial_energy_j:.3f}"
        )


def build_method_plans(system: SystemSpec, layout, ring) -> dict[str, ProtectionPlan]:
    """Build the four experiment methods: NP, OO, SO, AeroKV."""

    so_period = 32
    aerokv_fallback_period = 32
    plans: dict[str, ProtectionPlan] = {
        "NP": no_protection_plan(system, layout, ring),
        "OO": overlap_only_plan(system, layout, ring),
        "SO": snapshot_only_plan(system, layout, ring, period=so_period),
    }
    p1 = solve_p1_provisioning(system, layout, ring, method="AeroKV", beam_width=64)
    if p1.valid and p1.plan is not None:
        plans["AeroKV"] = p1.plan
    else:
        plans["AeroKV"] = make_fixed_protection_plan(
            system, layout, ring, method="AeroKV", overlap_depth=1, snapshot_period=aerokv_fallback_period
        )
    return plans


def shared_failure_trace(system: SystemSpec, seed: int, expected_failures_per_task: float) -> tuple[FailureEvent, ...]:
    return generate_poisson_failure_events(
        system,
        expected_failures_per_task=expected_failures_per_task,
        seed=seed + 991,
    )


def write_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())  # type: ignore[attr-defined]


def _format_result_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "NA"
    return str(value)


def print_result_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Print a compact fixed-width result table for an experiment."""

    if not rows:
        print(f"{title}: no results")
        return
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_format_result_value(row.get(col))))

    print(f"\n{title}")
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    for row in rows:
        print("  ".join(_format_result_value(row.get(col)).rjust(widths[col]) for col in columns))


def write_dict_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main_result_row(output) -> dict[str, object]:
    """Main result row used by experiment-level console output."""

    from aerokv.simulation.metrics import (
        cumulative_completion_time_s,
        final_token_uav_rows,
        mean_residual_energy_j,
    )

    final_uav_rows = final_token_uav_rows(output.uav_trace)
    return {
        "method": output.summary.method,
        "avg_remaining_energy_j": mean_residual_energy_j(final_uav_rows),
        "task_time_cost_s": cumulative_completion_time_s(output.token_trace),
        "predicted_remaining_tokens": output.summary.final_system_expected_remaining_tokens,
    }


def remaining_token_audit_row(output) -> dict[str, object]:
    """Compact final-state audit row for predicted remaining tokens."""

    from aerokv.simulation.metrics import (
        cumulative_completion_time_s,
        final_token_uav_rows,
        mean_residual_energy_j,
    )

    final_uav_rows = final_token_uav_rows(output.uav_trace)
    alive = [r for r in final_uav_rows if r.uav_status != "failed"]
    bottleneck = min(alive, key=lambda r: r.predicted_remaining_tokens_i) if alive else None
    return {
        "method": output.summary.method,
        "avg_remaining_energy_j": mean_residual_energy_j(final_uav_rows),
        "min_remaining_energy_j": min((r.energy_j for r in alive), default=0.0),
        "bottleneck_uav": None if bottleneck is None else bottleneck.uav_id,
        "future_energy_per_token_j_at_bottleneck": 0.0 if bottleneck is None else bottleneck.energy_per_token_j,
        "predicted_remaining_tokens": output.summary.final_system_expected_remaining_tokens,
        "task_time_cost_s": cumulative_completion_time_s(output.token_trace),
        "failure_status": output.summary.invalid_reason or ("success" if output.summary.mission_success else "incomplete"),
        "p1_new_valid": output.summary.p1_new_valid,
        "p2_valid": output.summary.p2_valid,
    }
