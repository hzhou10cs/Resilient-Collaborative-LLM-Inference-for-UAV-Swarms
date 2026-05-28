"""Compare P2 against uniform and no-reorg under heterogeneous UAV profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aerokv.config import make_heterogeneous_config
from aerokv.experiments._common import build_method_plans, build_standard_context, print_uav_profiles, shared_failure_trace
from aerokv.experiments.scenarios import contiguous_partition
from aerokv.optimizers.p1_new import solve_p1_new
from aerokv.optimizers.p2_reconfiguration import build_availability_index, p2_stage_latency_s, solve_p2_reconfiguration
from aerokv.protection_state import update_protection_runtime_at_completed_token
from aerokv.recovery import RecoveryResult, compute_recovery
from aerokv.specs import ExecutionLayout, LayerInterval, ProtectionPlan, SystemSpec


def _layout_text(layout: ExecutionLayout) -> str:
    return ", ".join(
        f"u{u}:[{iv.start},{iv.end})"
        for u, iv in sorted(layout.intervals.items(), key=lambda item: item[1].start)
    )


def _width_text(layout: ExecutionLayout) -> str:
    return ", ".join(
        f"u{u}:{iv.width}"
        for u, iv in sorted(layout.intervals.items(), key=lambda item: item[1].start)
    )


def _metrics_for_layout(system: SystemSpec, layout: ExecutionLayout, active: tuple[int, ...]) -> dict[str, object]:
    stages: list[float] = []
    for idx, uav_id in enumerate(active):
        next_uav = active[idx + 1] if idx + 1 < len(active) else None
        stage, _, _, _ = p2_stage_latency_s(system, uav_id, layout.interval(uav_id), next_uav)
        stages.append(stage)
    return {
        "total_chain_latency_s": sum(stages),
        "max_stage_latency_s": max(stages) if stages else 0.0,
        "stage_latencies_s": tuple(stages),
    }


def _uniform_layout(system: SystemSpec, active: tuple[int, ...]) -> ExecutionLayout:
    intervals = contiguous_partition(system.model.num_layers, len(active))
    return ExecutionLayout({uav_id: intervals[idx] for idx, uav_id in enumerate(active)})


def _no_reorg_summary(
    system: SystemSpec,
    plan: ProtectionPlan,
    active: tuple[int, ...],
    recovery: RecoveryResult,
) -> dict[str, object]:
    stages: list[float] = []
    pieces: list[str] = []
    widths: list[str] = []
    for idx, uav_id in enumerate(active):
        own = plan.layout.interval(uav_id).width if uav_id in plan.layout.intervals else 0
        recovered = sum(iv.width for iv in recovery.recovered_intervals_by_uav.get(uav_id, ()))
        width = own + recovered
        next_uav = active[idx + 1] if idx + 1 < len(active) else None
        stage, _, _, _ = p2_stage_latency_s(system, uav_id, LayerInterval(0, width), next_uav)
        stages.append(stage)
        widths.append(f"u{uav_id}:{width}")
        intervals = []
        if uav_id in plan.layout.intervals:
            iv = plan.layout.interval(uav_id)
            intervals.append(f"own[{iv.start},{iv.end})")
        for iv in recovery.recovered_intervals_by_uav.get(uav_id, ()):
            intervals.append(f"recovered[{iv.start},{iv.end})")
        pieces.append(f"u{uav_id}:" + "+".join(intervals))
    return {
        "partition": "; ".join(pieces),
        "widths": ", ".join(widths),
        "total_chain_latency_s": sum(stages),
        "max_stage_latency_s": max(stages) if stages else 0.0,
        "stage_latencies_s": tuple(stages),
    }


def run(seed: int = 2026, output_dir: str | Path = "outputs/exp1_heterogeneous_p2") -> list[dict[str, object]]:
    cfg = make_heterogeneous_config(seed)
    _, system, layout, ring = build_standard_context(seed, cfg)
    print_uav_profiles(system, cfg)
    plan = build_method_plans(system, layout, ring)["AeroKV"]
    failures = shared_failure_trace(system, seed, cfg.expected_failures_per_task)
    if not failures:
        raise RuntimeError("no failure generated for comparison")

    failure = failures[0]
    runtime = update_protection_runtime_at_completed_token(system, plan, failure.token).runtime
    alive_before = set(range(system.num_uavs))
    recovery = compute_recovery(system, plan, runtime, failure, alive_uavs_before_failure=alive_before)
    if not recovery.valid:
        raise RuntimeError(f"first recovery failed: {recovery.invalid_reason}")

    alive_after = alive_before - {failure.failed_uav}
    active = tuple(
        uav_id
        for uav_id, _ in sorted(plan.layout.intervals.items(), key=lambda item: item[1].start)
        if uav_id in alive_after
    )
    p2 = solve_p2_reconfiguration(
        system,
        plan,
        runtime,
        token=failure.token,
        alive_uavs=alive_after,
        recovered_intervals_by_uav=recovery.recovered_intervals_by_uav,
    )
    p1_new = solve_p1_new(system, p2, beam_width=128)

    rows: list[dict[str, object]] = []
    if p2.layout is not None:
        rows.append(
            {
                "method": "Selected P2",
                "valid": p2.valid,
                "p1_new_valid": p1_new.valid,
                "total_chain_latency_s": p2.total_chain_latency_s,
                "max_stage_latency_s": p2.max_stage_latency_s,
                "partition": _layout_text(p2.layout),
                "widths": _width_text(p2.layout),
                "stage_latencies_s": p2.stage_latencies_s,
            }
        )
    uniform = _uniform_layout(system, active)
    uniform_metrics = _metrics_for_layout(system, uniform, active)
    availability = build_availability_index(
        system,
        plan,
        runtime,
        token=failure.token,
        alive_uavs=alive_after,
        recovered_intervals_by_uav=recovery.recovered_intervals_by_uav,
    )
    uniform_feasible = all(availability.can_claim(u, uniform.interval(u)) for u in active)
    rows.append(
        {
            "method": "Uniform",
            "valid": uniform_feasible,
            "p1_new_valid": "",
            "total_chain_latency_s": uniform_metrics["total_chain_latency_s"],
            "max_stage_latency_s": uniform_metrics["max_stage_latency_s"],
            "partition": _layout_text(uniform),
            "widths": _width_text(uniform),
            "stage_latencies_s": uniform_metrics["stage_latencies_s"],
        }
    )
    no_reorg = _no_reorg_summary(system, plan, active, recovery)
    rows.append(
        {
            "method": "No-Reorg",
            "valid": True,
            "p1_new_valid": "",
            "total_chain_latency_s": no_reorg["total_chain_latency_s"],
            "max_stage_latency_s": no_reorg["max_stage_latency_s"],
            "partition": no_reorg["partition"],
            "widths": no_reorg["widths"],
            "stage_latencies_s": no_reorg["stage_latencies_s"],
        }
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "valid",
        "p1_new_valid",
        "total_chain_latency_s",
        "max_stage_latency_s",
        "widths",
        "partition",
        "stage_latencies_s",
    ]
    with (out / "p2_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[P2][comparison] failure_token={failure.token} failed_uav={failure.failed_uav} "
        f"p1_new_valid={p1_new.valid} p1_new_reason={p1_new.invalid_reason}"
    )
    for row in rows:
        print(
            f"[P2][comparison] method={row['method']} valid={row['valid']} "
            f"total_chain_latency_s={float(row['total_chain_latency_s']):.9f} "
            f"max_stage_latency_s={float(row['max_stage_latency_s']):.9f} widths={row['widths']}"
        )
    if p2.layout is not None and p2.total_chain_latency_s >= float(uniform_metrics["total_chain_latency_s"]):
        print(
            "[P2][comparison diagnostic] Selected P2 did not beat Uniform on total latency. "
            f"uniform_state_feasible={uniform_feasible}; state availability, survivor order, "
            "or memory constraints may prevent the unconstrained uniform comparison from being reachable."
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/exp1_heterogeneous_p2")
    args = parser.parse_args()
    run(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
