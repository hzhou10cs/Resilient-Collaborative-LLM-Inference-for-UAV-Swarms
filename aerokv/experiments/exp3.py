"""Experiment 3: ablation study.

Compares AeroKV-Full with simplified variants. The current implementation keeps
variant semantics explicit at the plan level; P2/P1-new integration in the
engine can be extended without changing this experiment interface.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import build_standard_context, shared_failure_trace
from aerokv.optimizers.p1_provisioning import solve_p1_provisioning
from aerokv.simulation.engine import make_fixed_protection_plan, simulate_fixed_plan_with_recovery


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp3") -> None:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    p1 = solve_p1_provisioning(system, layout, ring, method="AeroKV-Full", beam_width=64)
    full = p1.plan if p1.valid and p1.plan is not None else make_fixed_protection_plan(
        system, layout, ring, method="AeroKV-Full", overlap_depth=1, snapshot_period=128
    )
    variants = {
        "AeroKV-Full": full,
        "w/o P1-new": make_fixed_protection_plan(system, layout, ring, method="w/o P1-new", overlap_depth=1, snapshot_period=128),
        "w/o P2": make_fixed_protection_plan(system, layout, ring, method="w/o P2", overlap_depth=2, snapshot_period=128),
        "w/o P2+P1-new": make_fixed_protection_plan(system, layout, ring, method="w/o P2+P1-new", overlap_depth=1, snapshot_period=256),
    }
    rows = []
    for name, plan in variants.items():
        out = simulate_fixed_plan_with_recovery(
            system,
            plan,
            run_id=f"exp3-{name.replace(' ', '_')}-{seed}",
            failure_events=failures,
            expected_failures_per_task=cfg.expected_failures_per_task,
            print_progress=False,
        )
        rows.append(out.summary.to_dict())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp3")
    args = parser.parse_args()
    run(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
