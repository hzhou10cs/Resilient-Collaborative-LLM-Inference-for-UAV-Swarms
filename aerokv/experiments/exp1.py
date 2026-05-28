"""Experiment 1: overall end-to-end performance.

Matches the draft description: compare AeroKV against baselines under the same
Poisson attrition trace and report progress-time completion, residual energy,
and expected remaining token capacity inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerokv.config import ExperimentConfig
from aerokv.experiments._common import build_method_plans, build_standard_context, shared_failure_trace
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery, write_recovery_output


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp1", print_progress: bool = False) -> None:
    cfg = ExperimentConfig(seed=seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    plans = build_method_plans(system, layout, ring)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp1")
    parser.add_argument("--print-progress", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output_dir, args.print_progress)


if __name__ == "__main__":
    main()
