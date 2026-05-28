"""Single-threaded standard simulator with Poisson failures and recovery.

Layer 4 extends the fixed-plan simulator with:

- Poisson failure schedules with expected 2--3 failures per task by default.
- recovery latency and replay energy from the compact protection runtime.
- complete per-token step_log.csv.
- optional concise progress display every N tokens.

Still excluded: P2 reconfiguration, P1-new rebuilding, event log, RX energy,
bottleneck flags, KV segment debug, and multiprocessing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .accounting import (
    activation_buffer_tokens_for_source,
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    protected_pipeline_latency_s,
    snapshot_due_at_completed_token,
    snapshot_tail_layers_for_source,
    snapshot_tx_bytes_at_completed_token,
    tx_energy_for_bytes_j,
)
from .core import DroneRuntime, ProtectionPlan, ProtectionRuntime, RuntimeState, SystemSpec
from .failure_process import FailureEvent, format_failure_history, generate_poisson_failure_events
from .protection_runtime import update_protection_runtime_at_completed_token
from .recovery import RecoveryResult, compute_recovery
from .scenario import make_initial_layout, make_initial_ring, make_standard_system
from .sim_standard import make_fixed_protection_plan
from .trace_schema import StepLogRow, SummaryRow, TokenTraceRow, UAVTraceRow


@dataclass(frozen=True)
class RecoverySimulationOutput:
    summary: SummaryRow
    token_trace: list[TokenTraceRow]
    uav_trace: list[UAVTraceRow]
    step_log: list[StepLogRow]
    final_state: RuntimeState
    failure_events: tuple[FailureEvent, ...]
    recovery_results: tuple[RecoveryResult, ...]


# ---------------------------------------------------------------------------
# State helpers
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


def _alive_set(state: RuntimeState) -> set[int]:
    return {uav_id for uav_id, drone in state.drones.items() if drone.is_alive()}


def _freeze_failed_sources_runtime(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    previous: ProtectionRuntime,
    failed_sources: set[int],
) -> ProtectionRuntime:
    """Update protection runtime for alive sources, freeze failed sources."""

    full = update_protection_runtime_at_completed_token(system, plan, token).runtime
    latest: dict[int, int] = {}
    buffer: dict[int, int] = {}
    for source in plan.layout.intervals:
        if source in failed_sources:
            latest[source] = previous.latest_snapshot_token.get(source, 0)
            buffer[source] = previous.activation_buffer_tokens.get(source, 0)
        else:
            latest[source] = full.latest_snapshot_token.get(source, 0)
            buffer[source] = full.activation_buffer_tokens.get(source, 0)
    return ProtectionRuntime(latest_snapshot_token=latest, activation_buffer_tokens=buffer)


def _dynamic_exec_layers_by_uav(
    plan: ProtectionPlan,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
) -> dict[int, int]:
    layers: dict[int, int] = {uav_id: 0 for uav_id in plan.layout.intervals}
    for uav_id in alive:
        layers[uav_id] += plan.layout.interval(uav_id).width
        layers[uav_id] += recovered_exec_layers_by_uav.get(uav_id, 0)
    return layers


def _dynamic_protection_layers_by_uav(plan: ProtectionPlan, alive: set[int]) -> dict[int, int]:
    """Live-overlap compute layers still maintained without P1-new."""

    layers: dict[int, int] = {uav_id: 0 for uav_id in plan.layout.intervals}
    for source in alive:
        owner = plan.ring.pred(source)
        if owner in alive:
            layers[owner] += live_overlap_layers_for_source(plan, source)
    return layers


def _dynamic_stage_latency_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
) -> dict[int, float]:
    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    protection_layers = _dynamic_protection_layers_by_uav(plan, alive)
    out: dict[int, float] = {}
    for uav_id in alive:
        total_layers = exec_layers.get(uav_id, 0) + protection_layers.get(uav_id, 0)
        out[uav_id] = total_layers * system.uav(uav_id).per_layer_latency_s
    return out


def _dynamic_pipeline_latency_s(
    system: SystemSpec,
    plan: ProtectionPlan,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
) -> float:
    stage = _dynamic_stage_latency_by_uav(system, plan, alive, recovered_exec_layers_by_uav)
    return max(stage.values()) if stage else 0.0


def _snapshot_tx_bytes_for_alive_sources(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    alive: set[int],
) -> dict[int, int]:
    """Snapshot TX only occurs when both source and original successor are alive."""

    out: dict[int, int] = {}
    for source in alive:
        period = plan.snapshot_period[source]
        if period is None or not snapshot_due_at_completed_token(token, period):
            continue
        dest = plan.ring.succ(source)
        if dest not in alive:
            continue
        out[source] = snapshot_tx_bytes_at_completed_token(system, plan, source, token)
    return out


def _apply_token_energy(
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    *,
    pipeline_latency_s: float,
    recovered_exec_layers_by_uav: dict[int, int],
    snapshot_tx_bytes_by_source: dict[int, int],
) -> None:
    alive = _alive_set(state)
    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    protection_layers = _dynamic_protection_layers_by_uav(plan, alive)

    for uav_id in sorted(alive):
        drone = state.drones[uav_id]
        uav = system.uav(uav_id)
        flight_j = uav.flight_power_w * pipeline_latency_s
        compute_layers = exec_layers.get(uav_id, 0) + protection_layers.get(uav_id, 0)
        compute_j = uav.inference_power_w * compute_layers * uav.per_layer_latency_s
        tx_j = tx_energy_for_bytes_j(system, uav_id, snapshot_tx_bytes_by_source.get(uav_id, 0))

        drone.cumulative_flight_energy_j += flight_j
        drone.cumulative_compute_energy_j += compute_j
        drone.cumulative_tx_energy_j += tx_j
        drone.energy_j -= flight_j + compute_j + tx_j


def _apply_recovery_energy(
    system: SystemSpec,
    state: RuntimeState,
    result: RecoveryResult,
) -> None:
    """Apply flight energy during downtime and replay compute energy."""

    if result.recovery_latency_s <= 0:
        return
    for uav_id in sorted(_alive_set(state)):
        drone = state.drones[uav_id]
        flight_j = system.uav(uav_id).flight_power_w * result.recovery_latency_s
        drone.cumulative_flight_energy_j += flight_j
        drone.energy_j -= flight_j

    for uav_id, compute_j in result.replay_compute_energy_by_uav_j.items():
        if uav_id in state.drones and state.drones[uav_id].is_alive():
            state.drones[uav_id].cumulative_compute_energy_j += compute_j
            state.drones[uav_id].energy_j -= compute_j


def _dynamic_memory_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    token: int,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
) -> dict[int, float]:
    model = system.model
    out = {uav.uav_id: 0.0 for uav in system.uavs}

    for uav_id in alive:
        own_layers = plan.layout.interval(uav_id).width
        recovered_layers = recovered_exec_layers_by_uav.get(uav_id, 0)
        out[uav_id] += (own_layers + recovered_layers) * (
            model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer
        )

    # Live-overlap memory for alive sources whose original predecessor is alive.
    for source in alive:
        owner = plan.ring.pred(source)
        if owner not in alive:
            continue
        k = live_overlap_layers_for_source(plan, source)
        out[owner] += k * (model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer)

    # Snapshot memory and boundary activation buffer for alive sources whose
    # original successor is alive.  Failed sources are not P1-new protected here.
    for source in alive:
        owner = plan.ring.succ(source)
        if owner not in alive:
            continue
        tail_layers = snapshot_tail_layers_for_source(plan, source)
        if tail_layers == 0 or plan.snapshot_period[source] is None:
            continue
        latest = latest_snapshot_token_for_source(plan, source, token, runtime)
        latest = 0 if latest is None else latest
        out[owner] += tail_layers * latest * model.kv_bytes_per_token_layer
        out[owner] += activation_buffer_tokens_for_source(plan, source, token, runtime) * model.activation_bytes

    return out


def _refresh_dynamic_memory(
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    recovered_exec_layers_by_uav: dict[int, int],
) -> None:
    memory = _dynamic_memory_by_uav(
        system,
        plan,
        state.protection_runtime,
        state.token,
        _alive_set(state),
        recovered_exec_layers_by_uav,
    )
    for uav_id, mem in memory.items():
        state.drones[uav_id].memory_bytes = mem


# ---------------------------------------------------------------------------
# Trace/log helpers
# ---------------------------------------------------------------------------


def _failure_history_so_far(failures: list[FailureEvent]) -> str:
    return format_failure_history(tuple(failures))


def _make_token_row(
    run_id: str,
    method: str,
    system: SystemSpec,
    state: RuntimeState,
    pipeline_latency_s: float,
    failures_so_far: list[FailureEvent],
) -> TokenTraceRow:
    alive_drones = [d for d in state.drones.values() if d.is_alive()]
    return TokenTraceRow(
        run_id=run_id,
        method=method,
        token=state.token,
        time_s=state.time_s,
        phase=state.phase,
        failed_uav=state.failed_uav,
        num_alive_uavs=len(alive_drones),
        pipeline_latency_s=pipeline_latency_s,
        min_energy_j=state.min_alive_energy_j(),
        total_energy_j=sum(d.energy_j for d in alive_drones),
        total_memory_bytes=sum(d.memory_bytes for d in alive_drones),
        max_memory_bytes=max((d.memory_bytes for d in alive_drones), default=0.0),
        remaining_tokens=system.model.n_est - state.token,
        cumulative_compute_energy_j=sum(d.cumulative_compute_energy_j for d in state.drones.values()),
        cumulative_flight_energy_j=sum(d.cumulative_flight_energy_j for d in state.drones.values()),
        cumulative_tx_energy_j=sum(d.cumulative_tx_energy_j for d in state.drones.values()),
        num_failures=len(failures_so_far),
        failure_history=_failure_history_so_far(failures_so_far),
    )


def _snapshot_view_for_holder(
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    token: int,
    owner_uav: int,
    alive: set[int],
) -> tuple[int, int | None, int | None, int]:
    if owner_uav not in alive:
        return 0, None, None, 0
    source = plan.ring.pred(owner_uav)
    if source not in alive or plan.snapshot_period[source] is None:
        return 0, None, None, 0
    layers = snapshot_tail_layers_for_source(plan, source)
    latest = latest_snapshot_token_for_source(plan, source, token, runtime)
    stale = None if latest is None else token - latest
    buf = activation_buffer_tokens_for_source(plan, source, token, runtime)
    return layers, latest, stale, buf


def _make_uav_rows(
    run_id: str,
    method: str,
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    stage_latency_by_uav: dict[int, float],
    recovered_exec_layers_by_uav: dict[int, int],
) -> list[UAVTraceRow]:
    rows: list[UAVTraceRow] = []
    alive = _alive_set(state)
    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    protection_layers = _dynamic_protection_layers_by_uav(plan, alive)

    for uav_id in sorted(state.drones):
        drone = state.drones[uav_id]
        native = plan.layout.interval(uav_id)
        is_alive = drone.is_alive()
        snapshot_layers, latest, stale, buf = _snapshot_view_for_holder(
            plan, state.protection_runtime, state.token, uav_id, alive
        )
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
                native_layer_start=native.start if is_alive else None,
                native_layer_end=native.end if is_alive else None,
                exec_layer_start=native.start if is_alive else None,
                exec_layer_end=native.end if is_alive else None,
                num_exec_layers=exec_layers.get(uav_id, 0),
                live_overlap_layers=protection_layers.get(uav_id, 0),
                snapshot_layers=snapshot_layers,
                latest_snapshot_token=latest,
                snapshot_staleness_tokens=stale,
                activation_buffer_tokens=buf,
                stage_latency_s=stage_latency_by_uav.get(uav_id, 0.0),
                recovered_exec_layers=recovered_exec_layers_by_uav.get(uav_id, 0) if is_alive else 0,
            )
        )
    return rows


def _make_step_log_row(
    run_id: str,
    state: RuntimeState,
    failures_so_far: list[FailureEvent],
) -> StepLogRow:
    initial_total = sum(u.initial_energy_j for u in state.system.uavs)
    current_total = sum(d.energy_j for d in state.drones.values())
    avg_used = (initial_total - current_total) / max(1, state.system.num_uavs)
    return StepLogRow(
        run_id=run_id,
        token=state.token,
        time_s=state.time_s,
        phase=state.phase,
        num_alive_uavs=len(_alive_set(state)),
        avg_energy_used_j=avg_used,
        min_energy_j=state.min_alive_energy_j(),
        failure_history=_failure_history_so_far(failures_so_far),
    )


def _progress_line(row: StepLogRow) -> str:
    return (
        f"token={row.token} "
        f"time_s={row.time_s:.3f} "
        f"phase={row.phase} "
        f"alive={row.num_alive_uavs} "
        f"avg_energy_used_j={row.avg_energy_used_j:.3f} "
        f"min_energy_j={row.min_energy_j:.3f} "
        f"failures={row.failure_history}"
    )


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------


def simulate_fixed_plan_with_recovery(
    system: SystemSpec,
    plan: ProtectionPlan,
    *,
    run_id: str | None = None,
    failure_events: tuple[FailureEvent, ...] | None = None,
    expected_failures_per_task: float = 2.5,
    failure_seed: int | None = None,
    max_tokens: int | None = None,
    progress_interval_tokens: int = 20,
    print_progress: bool = False,
    output_dir: str | Path | None = None,
) -> RecoverySimulationOutput:
    """Run the standard fixed-plan simulation with failures and recovery.

    If ``failure_events`` is not provided, a Poisson failure schedule is sampled
    with expectation ``expected_failures_per_task`` over the task horizon.
    """

    plan.validate_against(system)
    if run_id is None:
        run_id = f"recovery-{uuid4().hex[:8]}"
    if max_tokens is None:
        max_tokens = system.model.n_est
    if not (1 <= max_tokens <= system.model.n_est):
        raise ValueError(f"max_tokens must be within [1, {system.model.n_est}]")
    if progress_interval_tokens <= 0:
        raise ValueError("progress_interval_tokens must be positive")

    if failure_events is None:
        failure_events = generate_poisson_failure_events(
            system,
            expected_failures_per_task=expected_failures_per_task,
            seed=failure_seed,
            max_token=max_tokens,
        )
    for event in failure_events:
        event.validate_against(system)
    failure_by_token: dict[int, list[FailureEvent]] = {}
    for event in sorted(failure_events, key=lambda e: e.token):
        if event.token <= max_tokens:
            failure_by_token.setdefault(event.token, []).append(event)

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

    recovered_exec_layers_by_uav: dict[int, int] = {}
    failed_sources: set[int] = set()
    failures_so_far: list[FailureEvent] = []
    recovery_results: list[RecoveryResult] = []
    memory_feasible = True
    energy_feasible = True
    deadline_met_all = True
    invalid_reason: str | None = None
    total_recovery_latency_s = 0.0
    total_protection_compute_energy_j = 0.0

    initial_latency = protected_pipeline_latency_s(system, plan).pipeline_latency_s
    _refresh_dynamic_memory(system, plan, state, recovered_exec_layers_by_uav)
    token_trace = [_make_token_row(run_id, plan.method, system, state, initial_latency, failures_so_far)]
    stage0 = _dynamic_stage_latency_by_uav(system, plan, _alive_set(state), recovered_exec_layers_by_uav)
    uav_trace = _make_uav_rows(run_id, plan.method, system, plan, state, stage0, recovered_exec_layers_by_uav)
    step_log = [_make_step_log_row(run_id, state, failures_so_far)]

    for token in range(1, max_tokens + 1):
        if invalid_reason is not None:
            break

        alive_before_token = _alive_set(state)
        pipeline_latency = _dynamic_pipeline_latency_s(system, plan, alive_before_token, recovered_exec_layers_by_uav)
        if pipeline_latency <= 0.0:
            invalid_reason = "no_alive_execution_stage"
            state.phase = "failed"
            break

        state.token = token
        state.time_s += pipeline_latency
        state.protection_runtime = _freeze_failed_sources_runtime(
            system, plan, token, state.protection_runtime, failed_sources
        )

        protection_layers_now = _dynamic_protection_layers_by_uav(plan, alive_before_token)
        total_protection_compute_energy_j += sum(
            system.uav(uav_id).inference_power_w * layers * system.uav(uav_id).per_layer_latency_s
            for uav_id, layers in protection_layers_now.items()
            if uav_id in alive_before_token
        )

        snapshot_tx = _snapshot_tx_bytes_for_alive_sources(system, plan, token, alive_before_token)
        _apply_token_energy(
            system,
            plan,
            state,
            pipeline_latency_s=pipeline_latency,
            recovered_exec_layers_by_uav=recovered_exec_layers_by_uav,
            snapshot_tx_bytes_by_source=snapshot_tx,
        )

        # Failures occur after the token has completed and before the next token.
        for event in failure_by_token.get(token, []):
            alive_before_failure = _alive_set(state)
            if event.failed_uav not in alive_before_failure:
                continue

            if recovered_exec_layers_by_uav.get(event.failed_uav, 0) > 0:
                failures_so_far.append(event)
                state.failed_uav = event.failed_uav
                state.drones[event.failed_uav].status = "failed"
                invalid_reason = "recovered_layers_owner_failed"
                deadline_met_all = False
                state.phase = "failed"
                break

            result = compute_recovery(
                system,
                plan,
                state.protection_runtime,
                event,
                alive_uavs_before_failure=alive_before_failure,
            )
            recovery_results.append(result)
            failures_so_far.append(event)
            state.failed_uav = event.failed_uav
            state.drones[event.failed_uav].status = "failed"
            state.phase = "recovering"

            if not result.valid:
                invalid_reason = result.invalid_reason or "recovery_failed"
                deadline_met_all = False
                state.phase = "failed"
                break

            total_recovery_latency_s += result.recovery_latency_s
            deadline_met_all = deadline_met_all and result.deadline_met
            state.time_s += result.recovery_latency_s
            _apply_recovery_energy(system, state, result)

            for owner, layers in result.recovered_exec_layers_by_uav.items():
                recovered_exec_layers_by_uav[owner] = recovered_exec_layers_by_uav.get(owner, 0) + layers
            failed_sources.add(event.failed_uav)
            state.phase = "post_failure"

        _refresh_dynamic_memory(system, plan, state, recovered_exec_layers_by_uav)

        for uav_id, drone in state.drones.items():
            if drone.is_alive() and drone.memory_bytes > system.uav(uav_id).memory_budget_bytes:
                memory_feasible = False
            if drone.energy_j < 0:
                energy_feasible = False
        if not memory_feasible and invalid_reason is None:
            invalid_reason = "memory_budget_exceeded"
            state.phase = "failed"
        if not energy_feasible and invalid_reason is None:
            invalid_reason = "energy_depleted"
            state.phase = "failed"

        alive_after = _alive_set(state)
        stage = _dynamic_stage_latency_by_uav(system, plan, alive_after, recovered_exec_layers_by_uav)
        next_latency = max(stage.values()) if stage else 0.0
        token_trace.append(_make_token_row(run_id, plan.method, system, state, next_latency, failures_so_far))
        uav_trace.extend(_make_uav_rows(run_id, plan.method, system, plan, state, stage, recovered_exec_layers_by_uav))
        log_row = _make_step_log_row(run_id, state, failures_so_far)
        step_log.append(log_row)

        if print_progress and (token % progress_interval_tokens == 0 or token == max_tokens):
            print(_progress_line(log_row))

    alive_drones = [d for d in state.drones.values() if d.is_alive()]
    terminal_min_energy = min((d.energy_j for d in alive_drones), default=0.0)
    completed = state.token == max_tokens and invalid_reason is None
    mission_success = completed and memory_feasible and energy_feasible and deadline_met_all
    if invalid_reason is None and not deadline_met_all:
        invalid_reason = "recovery_deadline_missed"

    first_failure = failures_so_far[0] if failures_so_far else None
    total_compute = sum(d.cumulative_compute_energy_j for d in state.drones.values())
    total_tx = sum(d.cumulative_tx_energy_j for d in state.drones.values())
    summary = SummaryRow(
        run_id=run_id,
        method=plan.method,
        seed=system.seed,
        num_uavs=system.num_uavs,
        num_layers=system.model.num_layers,
        n_est=system.model.n_est,
        failed_uav=None if first_failure is None else first_failure.failed_uav,
        failure_token=None if first_failure is None else first_failure.token,
        deadline_s=system.tau_recover_max_s,
        recovery_latency_s=total_recovery_latency_s,
        deadline_met=deadline_met_all,
        reconfiguration_latency_s=0.0,
        remaining_completion_time_s=0.0 if completed else max(0, system.model.n_est - state.token) * (_dynamic_pipeline_latency_s(system, plan, _alive_set(state), recovered_exec_layers_by_uav) if _alive_set(state) else 0.0),
        mission_complete_s=state.time_s if mission_success else None,
        terminal_min_energy_j=terminal_min_energy,
        protection_compute_energy_j=total_protection_compute_energy_j,
        tx_energy_j=total_tx,
        reconfiguration_energy_j=0.0,
        mission_success=mission_success,
        invalid_reason=invalid_reason,
        num_failures=len(failures_so_far),
        failure_trace=_failure_history_so_far(failures_so_far),
        expected_failures_per_task=expected_failures_per_task,
        total_recovery_latency_s=total_recovery_latency_s,
    )

    output = RecoverySimulationOutput(
        summary=summary,
        token_trace=token_trace,
        uav_trace=uav_trace,
        step_log=step_log,
        final_state=state,
        failure_events=failure_events,
        recovery_results=tuple(recovery_results),
    )
    if output_dir is not None:
        write_recovery_output(output, output_dir)
    return output


# ---------------------------------------------------------------------------
# Convenience entry point and CSV output
# ---------------------------------------------------------------------------


def run_standard_recovery(
    *,
    seed: int = 2026,
    overlap_depth: int = 1,
    snapshot_period: int | None = 128,
    expected_failures_per_task: float = 2.5,
    failure_seed: int | None = None,
    run_id: str | None = None,
    progress_interval_tokens: int = 20,
    print_progress: bool = False,
    output_dir: str | Path | None = None,
) -> RecoverySimulationOutput:
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
    return simulate_fixed_plan_with_recovery(
        system,
        plan,
        run_id=run_id,
        expected_failures_per_task=expected_failures_per_task,
        failure_seed=failure_seed,
        progress_interval_tokens=progress_interval_tokens,
        print_progress=print_progress,
        output_dir=output_dir,
    )


def _write_dataclass_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    names = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())  # type: ignore[attr-defined]


def write_recovery_output(output: RecoverySimulationOutput, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_dataclass_rows(out / "summary.csv", [output.summary], SummaryRow)
    _write_dataclass_rows(out / "token_trace.csv", output.token_trace, TokenTraceRow)
    _write_dataclass_rows(out / "uav_trace.csv", output.uav_trace, UAVTraceRow)
    _write_dataclass_rows(out / "step_log.csv", output.step_log, StepLogRow)


if __name__ == "__main__":
    result = run_standard_recovery(
        output_dir=Path("outputs/standard_recovery"),
        print_progress=True,
    )
    print(f"wrote recovery traces for run_id={result.summary.run_id}")
