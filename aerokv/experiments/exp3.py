"""Experiment 3: ablation study.

Compares AeroKV-Full with simplified variants and prints the main run metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import (
    build_standard_context,
    main_result_row,
    print_result_table,
    shared_failure_trace,
    write_dict_rows,
)
from aerokv.optimizers.p1_provisioning import solve_p1_provisioning
from aerokv.simulation.engine import make_fixed_protection_plan, simulate_fixed_plan_with_recovery

MAIN_COLUMNS = [
    "method",
    "avg_remaining_energy_j",
    "task_time_cost_s",
    "predicted_remaining_tokens",
    "mission_success",
]


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp3", print_progress: bool = True) -> list[dict[str, object]]:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    p1 = solve_p1_provisioning(system, layout, ring, method="AeroKV-Full", beam_width=64)
    full = p1.plan if p1.valid and p1.plan is not None else make_fixed_protection_plan(
        system, layout, ring, method="AeroKV-Full", overlap_depth=1, snapshot_period=32
    )
    variants = {
        "AeroKV-Full": full,
        "w/o P1-new": make_fixed_protection_plan(system, layout, ring, method="w/o P1-new", overlap_depth=1, snapshot_period=32),
        "w/o P2": make_fixed_protection_plan(system, layout, ring, method="w/o P2", overlap_depth=2, snapshot_period=32),
        "w/o P2+P1-new": make_fixed_protection_plan(system, layout, ring, method="w/o P2+P1-new", overlap_depth=1, snapshot_period=32),
    }
    rows: list[dict[str, object]] = []
    for name, plan in variants.items():
        out = simulate_fixed_plan_with_recovery(
            system,
            plan,
            run_id=f"exp3-{name.replace(' ', '_')}-{seed}",
            failure_events=failures,
            expected_failures_per_task=cfg.expected_failures_per_task,
            print_progress=print_progress,
        )
        row = main_result_row(out)
        row["mission_success"] = out.summary.mission_success
        rows.append(row)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dict_rows(out_dir / "ablation_summary.csv", rows, MAIN_COLUMNS)
    print_result_table("Experiment 3 main results", rows, MAIN_COLUMNS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp3")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output_dir, not args.no_progress)


if __name__ == "__main__":
    main()
