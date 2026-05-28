"""Experiment 1: overall end-to-end performance.

Compares AeroKV against baselines under the same Poisson attrition trace and
prints the main paper-facing metrics after the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import (
    build_method_plans,
    build_standard_context,
    main_result_row,
    print_result_table,
    shared_failure_trace,
    write_dict_rows,
)
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery, write_recovery_output

MAIN_COLUMNS = [
    "method",
    "avg_remaining_energy_j",
    "task_time_cost_s",
    "predicted_remaining_tokens",
]


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp1", print_progress: bool = True) -> list[dict[str, object]]:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    plans = build_method_plans(system, layout, ring)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    for method, plan in plans.items():
        out = simulate_fixed_plan_with_recovery(
            system,
            plan,
            run_id=f"exp1-{method.replace(' ', '_')}-{seed}",
            failure_events=failures,
            expected_failures_per_task=cfg.expected_failures_per_task,
            print_progress=print_progress,
            output_dir=out_root / method.replace(" ", "_"),
        )
        write_recovery_output(out, out_root / method.replace(" ", "_"))
        result_rows.append(main_result_row(out))

    write_dict_rows(out_root / "main_results.csv", result_rows, MAIN_COLUMNS)
    print_result_table("Experiment 1 main results", result_rows, MAIN_COLUMNS)
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp1")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output_dir, not args.no_progress)


if __name__ == "__main__":
    main()
