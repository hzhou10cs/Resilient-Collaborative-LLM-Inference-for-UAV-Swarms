"""Shared experiment helpers."""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from aerokv.baselines import full_mirror_plan, no_protection_plan, overlap_only_plan, snapshot_only_plan
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


def build_method_plans(system: SystemSpec, layout, ring) -> dict[str, ProtectionPlan]:
    periods = system.snapshot_period_candidates
    default_period = 128 if 128 in periods else periods[min(len(periods) - 1, 0)]
    plans: dict[str, ProtectionPlan] = {
        "NP": no_protection_plan(system, layout, ring),
        "OO": overlap_only_plan(system, layout, ring, k=1),
        "SO": snapshot_only_plan(system, layout, ring, period=default_period),
        "Ideal Full Mirror": full_mirror_plan(system, layout, ring),
    }
    p1 = solve_p1_provisioning(system, layout, ring, method="AeroKV", beam_width=64)
    if p1.valid and p1.plan is not None:
        plans["AeroKV"] = p1.plan
    else:
        plans["AeroKV"] = make_fixed_protection_plan(
            system, layout, ring, method="AeroKV", overlap_depth=1, snapshot_period=default_period
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
