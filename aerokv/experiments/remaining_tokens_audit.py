"""Diagnostics for predicted remaining token accounting."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from aerokv.config import ExperimentConfig, make_heterogeneous_config
from aerokv.experiments._common import (
    build_method_plans,
    build_standard_context,
    print_result_table,
    remaining_token_audit_row,
    shared_failure_trace,
    write_dict_rows,
)
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery, write_recovery_output

SEEDS = [0, 1, 2, 3, 10, 42, 100, 2026]

AUDIT_COLUMNS = [
    "seed",
    "method",
    "avg_remaining_energy_j",
    "min_remaining_energy_j",
    "bottleneck_uav",
    "future_energy_per_token_j_at_bottleneck",
    "predicted_remaining_tokens",
    "task_time_cost_s",
    "failure_status",
    "p1_new_valid",
    "p2_valid",
]

DETAIL_COLUMNS = [
    "seed",
    "method",
    "uav_id",
    "uav_status",
    "energy_j",
    "native_layer_start",
    "native_layer_end",
    "num_exec_layers",
    "recovered_exec_layers",
    "live_overlap_layers",
    "snapshot_layers",
    "inference_power_w",
    "tx_power_w",
    "overlap_power_w",
    "snapshot_boundary_power_w",
    "total_future_power_w",
    "per_token_latency_s",
    "token_rate",
    "energy_per_token_j",
    "predicted_remaining_tokens_i",
    "activation_forward_energy_per_token_j",
    "snapshot_boundary_energy_per_token_j",
]


def run(
    seeds: list[int] | None = None,
    output_dir: str | Path = "outputs/remaining_tokens_audit",
    *,
    heterogeneous: bool = False,
    print_progress: bool = False,
) -> list[dict[str, object]]:
    seeds = SEEDS if seeds is None else seeds
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    log_context = contextlib.nullcontext if print_progress else lambda: contextlib.redirect_stdout(io.StringIO())

    for seed in seeds:
        cfg = make_heterogeneous_config(seed=seed) if heterogeneous else ExperimentConfig(seed=seed)
        _, system, layout, ring = build_standard_context(seed, cfg)
        failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
        with log_context():
            plans = build_method_plans(system, layout, ring)
        for method, plan in plans.items():
            method_dir = out_root / f"seed_{seed}" / method.replace(" ", "_")
            with log_context():
                out = simulate_fixed_plan_with_recovery(
                    system,
                    plan,
                    run_id=f"remaining-token-audit-{method}-{seed}",
                    failure_events=failures,
                    expected_failures_per_task=cfg.expected_failures_per_task,
                    print_progress=print_progress,
                    output_dir=method_dir,
                )
            write_recovery_output(out, method_dir)
            row = {"seed": seed}
            row.update(remaining_token_audit_row(out))
            audit_rows.append(row)

            final_token = max((r.token for r in out.uav_trace), default=0)
            for r in out.uav_trace:
                if r.token != final_token:
                    continue
                detail_rows.append(
                    {
                        "seed": seed,
                        **{name: getattr(r, name) for name in DETAIL_COLUMNS if name not in {"seed"}},
                    }
                )

    write_dict_rows(out_root / "seed_sweep.csv", audit_rows, AUDIT_COLUMNS)
    write_dict_rows(out_root / "final_uav_diagnostics.csv", detail_rows, DETAIL_COLUMNS)
    print_result_table("Predicted remaining token seed sweep", audit_rows, AUDIT_COLUMNS)
    return audit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/remaining_tokens_audit")
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    run(args.seeds, args.output_dir, heterogeneous=args.heterogeneous, print_progress=args.progress)


if __name__ == "__main__":
    main()
