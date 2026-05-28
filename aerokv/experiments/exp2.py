"""Experiment 2: recovery communication overhead comparison.

Produces per-method summary rows for protected-state memory, snapshot traffic,
recovery latency, and recovery energy proxies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import build_method_plans, build_standard_context, shared_failure_trace
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp2") -> None:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    rows = []
    for method, plan in build_method_plans(system, layout, ring).items():
        out = simulate_fixed_plan_with_recovery(
            system,
            plan,
            run_id=f"exp2-{method.replace(' ', '_')}-{seed}",
            failure_events=failures,
            expected_failures_per_task=cfg.expected_failures_per_task,
            print_progress=False,
        )
        max_memory = max(r.max_memory_bytes for r in out.token_trace)
        rows.append(
            {
                "method": method,
                "max_protected_state_memory_bytes": max_memory,
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
    with (out_dir / "overhead_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp2")
    args = parser.parse_args()
    run(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
