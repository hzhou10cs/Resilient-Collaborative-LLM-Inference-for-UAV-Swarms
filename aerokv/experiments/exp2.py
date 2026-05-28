"""Experiment 2: recovery/protection overhead comparison.

Prints the main overhead metrics after running each method under the same
failure trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import (
    build_method_plans,
    build_standard_context,
    print_result_table,
    shared_failure_trace,
    write_dict_rows,
)
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery

MAIN_COLUMNS = [
    "method",
    "max_protected_state_memory_bytes",
    "snapshot_tx_energy_j",
    "total_recovery_latency_s",
    "deadline_met",
    "mission_success",
]


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp2", print_progress: bool = True) -> list[dict[str, object]]:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    rows: list[dict[str, object]] = []
    for method, plan in build_method_plans(system, layout, ring).items():
        out = simulate_fixed_plan_with_recovery(
            system,
            plan,
            run_id=f"exp2-{method.replace(' ', '_')}-{seed}",
            failure_events=failures,
            expected_failures_per_task=cfg.expected_failures_per_task,
            print_progress=print_progress,
        )
        rows.append(
            {
                "method": method,
                "max_protected_state_memory_bytes": max(r.max_memory_bytes for r in out.token_trace),
                "snapshot_tx_energy_j": out.summary.tx_energy_j,
                "total_recovery_latency_s": out.summary.total_recovery_latency_s,
                "deadline_met": out.summary.deadline_met,
                "mission_success": out.summary.mission_success,
                "num_failures": out.summary.num_failures,
                "failure_trace": out.summary.failure_trace,
            }
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dict_rows(out_dir / "overhead_table.csv", rows, list(rows[0]))
    print_result_table("Experiment 2 main results", rows, MAIN_COLUMNS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp2")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output_dir, not args.no_progress)


if __name__ == "__main__":
    main()
