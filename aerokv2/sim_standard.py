"""Single-threaded standard fixed-plan simulator.

Layer 3 adds the first simulator loop.  It is intentionally narrow:

- fixed protection plan only
- pre-failure generation only
- per-token system trace
- per-token-per-UAV trace
- no recovery
- no reconfiguration
- no P1/P2 planner
- no event log
- no RX energy
- no bottleneck flags
- no multiprocessing

Token convention:

- token == 0 is the initial state before generation.
- token == t > 0 is the state after completing t generated tokens.
- time_s is the cumulative logical mission time at that completed token.
  Since this layer has no failure/recovery downtime, time_s = token *
  pipeline_latency_s.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .accounting import (
    memory_by_uav_bytes,
    protected_compute_energy_per_token_j,
    protected_pipeline_latency_s,
    protection_compute_energy_per_token_j,
    tx_energy_for_bytes_j,
)
from .core import DroneRuntime, ExecutionLayout, LogicalRing, ProtectionPlan, RuntimeState, SystemSpec
from .protection_runtime import owner_protection_view, update_protection_runtime_at_completed_token
from .scenario import make_initial_layout, make_initial_ring, make_standard_system
from .trace_schema import SummaryRow, TokenTraceRow, UAVTraceRow


@dataclass(frozen=True)
class StandardSimulationOutput:
    """In-memory result for one fixed-plan standard run."""

    summary: SummaryRow
    token_trace: list[TokenTraceRow]
    uav_trace: list[UAVTraceRow]
    final_state: RuntimeState


# ---------------------------------------------------------------------------
# Fixed plan construction
# ---------------------------------------------------------------------------


def make_fixed_protection_plan(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    method: str = "AeroKV",
    overlap_depth: int = 1,
    snapshot_period: int | None = 128,
    full_mirror: bool = False,
) -> ProtectionPlan:
    """Build a simple fixed protection plan for simulator bring-up.

    This is not P1.  It applies the same overlap depth and snapshot period to
    every source UAV, clamping overlap depth to each shard width.
    """

    if overlap_depth < 0:
        raise ValueError("overlap_depth must be non-negative")
    if snapshot_period is not None and snapshot_period <= 0:
        raise ValueError("snapshot_period must be positive or None")

    head_overlap_depth = {
        uav_id: min(overlap_depth, layout.interval(uav_id).width)
        for uav_id in layout.intervals.keys()
    }
    snapshot_periods = {uav_id: snapshot_period for uav_id in layout.intervals.keys()}

    plan = ProtectionPlan(
        method=method,
        layout=layout,
        ring=ring,
        head_overlap_depth=head_overlap_depth,
        snapshot_period=snapshot_periods,
        full_mirror=full_mirror,
    )
    plan.validate_against(system)
    return plan


def make_standard_fixed_plan(
    *,
    seed: int = 2026,
    overlap_depth: int = 1,
    snapshot_period: int | None = 128,
) -> tuple[SystemSpec, ProtectionPlan]:
    """Construct the standard system and a fixed AeroKV protection plan."""

    system = make_standard_system(seed=seed)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    plan = make_fixed_protection_plan(
        system,
        layout,
        ring,
        method="AeroKV",
        overlap_depth=overlap_depth,
        snapshot_period=snapshot_period,
    )
    return system, plan


# ---------------------------------------------------------------------------
# Trace row construction
# ---------------------------------------------------------------------------


def _initial_drone_runtime(system: SystemSpec) -> dict[int, DroneRuntime]:
    return {
        uav.uav_id: DroneRuntime(
            uav_id=uav.uav_id,
            status="normal",
            energy_j=uav.initial_energy_j,
        )
        for uav in system.uavs
    }


def _update_energy_for_completed_token(
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    snapshot_tx_bytes_by_source: dict[int, int],
    pipeline_latency_s: float,
) -> None:
    """Apply one token of flight, compute, and exact snapshot TX energy."""

    for uav_id in sorted(state.drones):
        drone = state.drones[uav_id]
        if not drone.is_alive():
            continue

        uav = system.uav(uav_id)
        flight_j = uav.flight_power_w * pipeline_latency_s
        compute_j = protected_compute_energy_per_token_j(system, plan, uav_id)
        tx_j = tx_energy_for_bytes_j(system, uav_id, snapshot_tx_bytes_by_source.get(uav_id, 0))

        drone.cumulative_flight_energy_j += flight_j
        drone.cumulative_compute_energy_j += compute_j
        drone.cumulative_tx_energy_j += tx_j
        drone.energy_j -= flight_j + compute_j + tx_j


def _refresh_memory(system: SystemSpec, plan: ProtectionPlan, state: RuntimeState, token: int) -> None:
    memory = memory_by_uav_bytes(system, plan, token, state.protection_runtime)
    for uav_id, memory_bytes in memory.items():
        state.drones[uav_id].memory_bytes = memory_bytes


def _make_token_row(
    run_id: str,
    method: str,
    system: SystemSpec,
    state: RuntimeState,
    pipeline_latency_s: float,
) -> TokenTraceRow:
    alive = [d for d in state.drones.values() if d.is_alive()]
    total_energy = sum(d.energy_j for d in alive)
    total_memory = sum(d.memory_bytes for d in alive)
    max_memory = max((d.memory_bytes for d in alive), default=0.0)

    return TokenTraceRow(
        run_id=run_id,
        method=method,
        token=state.token,
        time_s=state.time_s,
        phase=state.phase,
        failed_uav=state.failed_uav,
        num_alive_uavs=len(alive),
        pipeline_latency_s=pipeline_latency_s,
        min_energy_j=state.min_alive_energy_j(),
        total_energy_j=total_energy,
        total_memory_bytes=total_memory,
        max_memory_bytes=max_memory,
        remaining_tokens=system.model.n_est - state.token,
        cumulative_compute_energy_j=sum(d.cumulative_compute_energy_j for d in alive),
        cumulative_flight_energy_j=sum(d.cumulative_flight_energy_j for d in alive),
        cumulative_tx_energy_j=sum(d.cumulative_tx_energy_j for d in alive),
    )


def _make_uav_rows(
    run_id: str,
    method: str,
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    stage_latency_by_uav: dict[int, float],
) -> list[UAVTraceRow]:
    rows: list[UAVTraceRow] = []
    for uav_id in sorted(state.drones):
        drone = state.drones[uav_id]
        native = plan.layout.interval(uav_id)
        owner_view = owner_protection_view(system, plan, uav_id, state.token, state.protection_runtime)

        rows.append(
            UAVTraceRow(
                run_id=run_id,
                method=method,
                token=state.token,
                time_s=state.time_s,
                uav_id=uav_id,
                uav_status=drone.status,
                energy_j=drone.energy_j,
                flight_energy_j=drone.cumulative_flight_energy_j,
                compute_energy_j=drone.cumulative_compute_energy_j,
                tx_energy_j=drone.cumulative_tx_energy_j,
                memory_bytes=drone.memory_bytes,
                native_layer_start=native.start,
                native_layer_end=native.end,
                exec_layer_start=native.start,
                exec_layer_end=native.end,
                num_exec_layers=native.width,
                live_overlap_layers=owner_view.live_overlap_layers,
                snapshot_layers=owner_view.snapshot_layers,
                latest_snapshot_token=owner_view.latest_snapshot_token,
                snapshot_staleness_tokens=owner_view.snapshot_staleness_tokens,
                activation_buffer_tokens=owner_view.activation_buffer_tokens,
                stage_latency_s=stage_latency_by_uav.get(uav_id, 0.0),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Main fixed-plan simulator
# ---------------------------------------------------------------------------


def simulate_fixed_plan_prefailure(
    system: SystemSpec,
    plan: ProtectionPlan,
    *,
    run_id: str | None = None,
    max_tokens: int | None = None,
    output_dir: str | Path | None = None,
) -> StandardSimulationOutput:
    """Run a single-threaded pre-failure fixed-plan standard simulation.

    The returned traces include token 0 initial state and one row for every
    completed generated token up to ``max_tokens`` or ``system.model.n_est``.
    """

    plan.validate_against(system)
    if run_id is None:
        run_id = f"standard-{uuid4().hex[:8]}"
    if max_tokens is None:
        max_tokens = system.model.n_est
    if not (0 <= max_tokens <= system.model.n_est):
        raise ValueError(f"max_tokens must be within [0, {system.model.n_est}]")

    latency = protected_pipeline_latency_s(system, plan)
    pipeline_latency = latency.pipeline_latency_s
    stage_latency_by_uav = dict(latency.stage_latency_by_uav)

    state = RuntimeState(
        token=0,
        time_s=0.0,
        phase="pre_failure",
        system=system,
        layout=plan.layout,
        ring=plan.ring,
        protection_plan=plan,
        protection_runtime=update_protection_runtime_at_completed_token(system, plan, 0).runtime,
        drones=_initial_drone_runtime(system),
        failed_uav=None,
    )
    _refresh_memory(system, plan, state, token=0)

    token_trace: list[TokenTraceRow] = [
        _make_token_row(run_id, plan.method, system, state, pipeline_latency)
    ]
    uav_trace: list[UAVTraceRow] = _make_uav_rows(
        run_id,
        plan.method,
        system,
        plan,
        state,
        stage_latency_by_uav,
    )

    memory_feasible_all_tokens = True
    energy_feasible_all_tokens = True

    for token in range(1, max_tokens + 1):
        update = update_protection_runtime_at_completed_token(system, plan, token)
        state.token = token
        state.time_s = token * pipeline_latency
        state.protection_runtime = update.runtime

        _update_energy_for_completed_token(
            system,
            plan,
            state,
            dict(update.snapshot_tx_bytes_by_source),
            pipeline_latency,
        )
        _refresh_memory(system, plan, state, token=token)

        for uav_id, drone in state.drones.items():
            if drone.memory_bytes > system.uav(uav_id).memory_budget_bytes:
                memory_feasible_all_tokens = False
            if drone.is_alive() and drone.energy_j < 0:
                energy_feasible_all_tokens = False

        token_trace.append(_make_token_row(run_id, plan.method, system, state, pipeline_latency))
        uav_trace.extend(_make_uav_rows(run_id, plan.method, system, plan, state, stage_latency_by_uav))

    alive_drones = [d for d in state.drones.values() if d.is_alive()]
    terminal_min_energy = min((d.energy_j for d in alive_drones), default=0.0)
    mission_success = memory_feasible_all_tokens and energy_feasible_all_tokens and max_tokens == system.model.n_est
    invalid_reason = None
    if not memory_feasible_all_tokens:
        invalid_reason = "memory_budget_exceeded"
    elif not energy_feasible_all_tokens:
        invalid_reason = "energy_depleted"
    elif max_tokens != system.model.n_est:
        invalid_reason = "partial_run"

    summary = SummaryRow(
        run_id=run_id,
        method=plan.method,
        seed=system.seed,
        num_uavs=system.num_uavs,
        num_layers=system.model.num_layers,
        n_est=system.model.n_est,
        failed_uav=None,
        failure_token=None,
        deadline_s=system.tau_recover_max_s,
        recovery_latency_s=0.0,
        deadline_met=True,
        reconfiguration_latency_s=0.0,
        remaining_completion_time_s=0.0,
        mission_complete_s=state.time_s if mission_success else None,
        terminal_min_energy_j=terminal_min_energy,
        protection_compute_energy_j=sum(d.cumulative_compute_energy_j for d in alive_drones)
        - sum(
            system.uav(uav_id).inference_power_w
            * plan.layout.interval(uav_id).width
            * system.uav(uav_id).per_layer_latency_s
            * max_tokens
            for uav_id in plan.layout.intervals
        ),
        tx_energy_j=sum(d.cumulative_tx_energy_j for d in alive_drones),
        reconfiguration_energy_j=0.0,
        mission_success=mission_success,
        invalid_reason=invalid_reason,
    )

    output = StandardSimulationOutput(
        summary=summary,
        token_trace=token_trace,
        uav_trace=uav_trace,
        final_state=state,
    )
    if output_dir is not None:
        write_standard_output(output, output_dir)
    return output


def run_standard_fixed_plan(
    *,
    seed: int = 2026,
    overlap_depth: int = 1,
    snapshot_period: int | None = 128,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> StandardSimulationOutput:
    """Convenience entry point for the standard model fixed-plan run."""

    system, plan = make_standard_fixed_plan(
        seed=seed,
        overlap_depth=overlap_depth,
        snapshot_period=snapshot_period,
    )
    return simulate_fixed_plan_prefailure(system, plan, run_id=run_id, output_dir=output_dir)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_dataclass_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    names = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())  # type: ignore[attr-defined]


def write_standard_output(output: StandardSimulationOutput, output_dir: str | Path) -> None:
    """Write summary.csv, token_trace.csv, and uav_trace.csv."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_dataclass_rows(out / "summary.csv", [output.summary], SummaryRow)
    _write_dataclass_rows(out / "token_trace.csv", output.token_trace, TokenTraceRow)
    _write_dataclass_rows(out / "uav_trace.csv", output.uav_trace, UAVTraceRow)


# Backward-compatible concise names used by the layer tests.
def run_standard_pre_failure(
    system: SystemSpec,
    plan: ProtectionPlan,
    *,
    run_id: str | None = None,
    max_tokens: int | None = None,
    output_dir: str | Path | None = None,
) -> StandardSimulationOutput:
    return simulate_fixed_plan_prefailure(
        system,
        plan,
        run_id=run_id,
        max_tokens=max_tokens,
        output_dir=output_dir,
    )


def write_standard_outputs(output: StandardSimulationOutput, output_dir: str | Path) -> None:
    write_standard_output(output, output_dir)



if __name__ == "__main__":
    result = run_standard_fixed_plan(output_dir=Path("outputs/standard_fixed_plan"))
    print(f"wrote standard fixed-plan traces for run_id={result.summary.run_id}")
