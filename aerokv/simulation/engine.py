"""Single-threaded AeroKV experiment engine.

This is the production simulation loop used by the experiment scripts.  It
supports method-specific recovery for NP/OO/SO/AeroKV and, for AeroKV, performs
post-failure P2 reconfiguration followed by P1-new protection rebuilding.

Modeling choices kept intentionally simple:
- live-overlap compute is accounted as energy but hidden from main decode
  latency under the simplified bubble model;
- P2/P1-new planner control time is not counted as compute latency;
- P1-new rebuild data movement is simplified as a fixed 200 J reconfiguration
  energy overhead and zero planner latency;
- communication energy is TX-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from ..accounting import (
    activation_buffer_tokens_for_source,
    average_snapshot_tx_bytes_per_token,
    effective_snapshot_period,
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    protected_pipeline_latency_s,
    runtime_inference_power_w,
    runtime_per_layer_latency_s,
    snapshot_due_at_completed_token,
    snapshot_tail_layers_for_source,
    snapshot_tx_bytes_at_completed_token,
    tx_energy_for_bytes_j,
)
from ..experiments.scenarios import make_initial_layout, make_initial_ring, make_standard_system
from ..optimizers.p1_new import solve_p1_new
from ..optimizers.p2_reconfiguration import solve_p2_reconfiguration
from ..protection_state import update_protection_runtime_at_completed_token
from ..recovery import RecoveryResult, compute_recovery
from ..specs import (
    DroneRuntime,
    ExecutionLayout,
    LogicalRing,
    ProtectionPlan,
    ProtectionRuntime,
    RuntimeState,
    SystemSpec,
)
from .events import FailureEvent, format_failure_history, generate_poisson_failure_events
from .traces import StepLogRow, SummaryRow, TokenTraceRow, UAVTraceRow


@dataclass(frozen=True)
class RecoverySimulationOutput:
    summary: SummaryRow
    token_trace: list[TokenTraceRow]
    uav_trace: list[UAVTraceRow]
    step_log: list[StepLogRow]
    final_state: RuntimeState
    failure_events: tuple[FailureEvent, ...]
    recovery_results: tuple[RecoveryResult, ...]


@dataclass(frozen=True)
class FutureTokenDiagnostics:
    uav_id: int
    remaining_energy_j: float
    exec_layers: int
    overlap_layers: int
    inference_power_w: float
    tx_power_w: float
    overlap_power_w: float
    snapshot_boundary_power_w: float
    total_future_power_w: float
    per_token_latency_s: float
    token_rate: float
    execution_compute_energy_j: float
    overlap_compute_energy_j: float
    flight_energy_j: float
    snapshot_boundary_energy_j: float
    activation_forward_energy_j: float
    energy_per_token_j: float
    predicted_remaining_tokens_i: float


# ---------------------------------------------------------------------------
# Fixed plan construction helpers retained for experiment/backward compatibility
# ---------------------------------------------------------------------------


def make_fixed_protection_plan(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    method: str = "AeroKV",
    overlap_depth: int = 1,
    snapshot_period: int | None = 1,
) -> ProtectionPlan:
    if overlap_depth < 0:
        raise ValueError("overlap_depth must be non-negative")
    if snapshot_period is not None and snapshot_period <= 0:
        raise ValueError("snapshot_period must be positive or None")

    plan = ProtectionPlan(
        method=method,
        layout=layout,
        ring=ring,
        head_overlap_depth={u: min(overlap_depth, layout.interval(u).width) for u in layout.intervals},
        snapshot_period={u: snapshot_period for u in layout.intervals},
    )
    plan.validate_against(system)
    return plan


def make_standard_fixed_plan(
    *,
    seed: int = 2026,
    overlap_depth: int = 1,
    snapshot_period: int | None = 1,
) -> tuple[SystemSpec, ProtectionPlan]:
    system = make_standard_system(seed=seed)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    return system, make_fixed_protection_plan(
        system,
        layout,
        ring,
        method="AeroKV",
        overlap_depth=overlap_depth,
        snapshot_period=snapshot_period,
    )


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _initial_drone_runtime(system: SystemSpec) -> dict[int, DroneRuntime]:
    return {
        uav.uav_id: DroneRuntime(uav_id=uav.uav_id, status="normal", energy_j=uav.initial_energy_j)
        for uav in system.uavs
    }


def _alive_set(state: RuntimeState) -> set[int]:
    return {uav_id for uav_id, drone in state.drones.items() if drone.is_alive()}


def _method_is_aerokv(plan: ProtectionPlan) -> bool:
    return plan.method.strip().upper().startswith("AEROKV")


def _fresh_runtime_for_plan_at_token(system: SystemSpec, plan: ProtectionPlan, token: int) -> ProtectionRuntime:
    """Create a fresh protection runtime after P1-new rebuilding.

    The current simulator does not yet model P1-new rebuild latency/data movement.
    We therefore treat the rebuilt protection state as becoming available at the
    current token: snapshot tails are fresh at ``token`` and activation buffers
    start empty.
    """

    latest: dict[int, int] = {}
    buffers: dict[int, int] = {}
    for source in plan.layout.intervals:
        if effective_snapshot_period(plan, source) is not None and snapshot_tail_layers_for_source(plan, source) > 0:
            latest[source] = token
            buffers[source] = 0
    return ProtectionRuntime(latest_snapshot_token=latest, activation_buffer_tokens=buffers)


def _freeze_failed_sources_runtime(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    previous: ProtectionRuntime,
    failed_sources: set[int],
) -> ProtectionRuntime:
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
        if uav_id not in plan.layout.intervals:
            continue
        layers[uav_id] += plan.layout.interval(uav_id).width
        layers[uav_id] += recovered_exec_layers_by_uav.get(uav_id, 0)
    return layers


def _dynamic_protection_layers_by_uav(plan: ProtectionPlan, alive: set[int]) -> dict[int, int]:
    layers: dict[int, int] = {uav_id: 0 for uav_id in plan.layout.intervals}
    for source in alive:
        if source not in plan.layout.intervals:
            continue
        owner = plan.ring.pred(source)
        if owner in alive and owner in layers:
            layers[owner] += live_overlap_layers_for_source(plan, source)
    return layers


def _dynamic_stage_latency_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
    energy_by_uav_j: dict[int, float] | None = None,
) -> dict[int, float]:
    """Decode per-token stage latencies with live-overlap hidden in bubbles."""

    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    out: dict[int, float] = {}
    for u in alive:
        if u not in plan.layout.intervals:
            continue
        energy_j = None if energy_by_uav_j is None else energy_by_uav_j.get(u)
        out[u] = exec_layers.get(u, 0) * runtime_per_layer_latency_s(system, u, energy_j)
    return out

def _dynamic_pipeline_latency_s(
    system: SystemSpec,
    plan: ProtectionPlan,
    alive: set[int],
    recovered_exec_layers_by_uav: dict[int, int],
    energy_by_uav_j: dict[int, float] | None = None,
) -> float:
    stage = _dynamic_stage_latency_by_uav(system, plan, alive, recovered_exec_layers_by_uav, energy_by_uav_j)
    return sum(stage.values()) if stage else 0.0

def _snapshot_tx_bytes_for_alive_sources(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    alive: set[int],
) -> dict[int, int]:
    out: dict[int, int] = {}
    for source in alive:
        if source not in plan.layout.intervals:
            continue
        period = effective_snapshot_period(plan, source)
        if period is None or not snapshot_due_at_completed_token(token, period):
            continue
        dest = plan.ring.succ(source)
        if dest not in alive:
            continue
        out[source] = snapshot_tx_bytes_at_completed_token(system, plan, source, token)
    return out


def _average_snapshot_tx_bytes_for_alive_sources(
    system: SystemSpec,
    plan: ProtectionPlan,
    alive: set[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for source in alive:
        if source not in plan.layout.intervals:
            continue
        dest = plan.ring.succ(source)
        if dest not in alive:
            continue
        bytes_per_token = average_snapshot_tx_bytes_per_token(system, plan, source)
        if bytes_per_token > 0.0:
            out[source] = bytes_per_token
    return out


def _activation_forward_sources(plan: ProtectionPlan, alive: set[int]) -> set[int]:
    executing = [
        uav_id
        for uav_id, interval in sorted(plan.layout.intervals.items(), key=lambda item: item[1].start)
        if uav_id in alive and interval.width > 0
    ]
    return set(executing[:-1])


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
        if uav_id not in plan.layout.intervals:
            continue
        drone = state.drones[uav_id]
        uav = system.uav(uav_id)
        flight_j = uav.flight_power_w * pipeline_latency_s
        compute_layers = exec_layers.get(uav_id, 0) + protection_layers.get(uav_id, 0)
        compute_j = runtime_inference_power_w(system, uav_id, state.drones[uav_id].energy_j) * compute_layers * runtime_per_layer_latency_s(system, uav_id, state.drones[uav_id].energy_j)
        tx_j = tx_energy_for_bytes_j(system, uav_id, snapshot_tx_bytes_by_source.get(uav_id, 0))

        drone.cumulative_flight_energy_j += flight_j
        drone.cumulative_compute_energy_j += compute_j
        drone.cumulative_tx_energy_j += tx_j
        drone.energy_j -= flight_j + compute_j + tx_j


def _apply_recovery_energy(system: SystemSpec, state: RuntimeState, result: RecoveryResult) -> None:
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
        if uav_id not in plan.layout.intervals:
            continue
        own_layers = plan.layout.interval(uav_id).width
        recovered_layers = recovered_exec_layers_by_uav.get(uav_id, 0)
        out[uav_id] += (own_layers + recovered_layers) * (
            model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer
        )

    for source in alive:
        if source not in plan.layout.intervals:
            continue
        owner = plan.ring.pred(source)
        if owner not in alive:
            continue
        k = live_overlap_layers_for_source(plan, source)
        out[owner] += k * (model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer)

    for source in alive:
        if source not in plan.layout.intervals:
            continue
        owner = plan.ring.succ(source)
        if owner not in alive:
            continue
        tail_layers = snapshot_tail_layers_for_source(plan, source)
        if tail_layers == 0 or effective_snapshot_period(plan, source) is None:
            continue
        latest = latest_snapshot_token_for_source(plan, source, token, runtime)
        latest = 0 if latest is None else latest
        out[owner] += tail_layers * latest * model.kv_bytes_per_token_layer

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


def _future_token_diagnostics_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    pipeline_latency_s: float,
    recovered_exec_layers_by_uav: dict[int, int],
) -> dict[int, FutureTokenDiagnostics]:
    alive = _alive_set(state)
    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    protection_layers = _dynamic_protection_layers_by_uav(plan, alive)
    snapshot_tx = _average_snapshot_tx_bytes_for_alive_sources(system, plan, alive)
    activation_forward_sources = _activation_forward_sources(plan, alive)

    out: dict[int, FutureTokenDiagnostics] = {}
    for uav_id in alive:
        if uav_id not in plan.layout.intervals:
            continue
        uav = system.uav(uav_id)
        inference_power = runtime_inference_power_w(system, uav_id, state.drones[uav_id].energy_j)
        per_layer_latency = runtime_per_layer_latency_s(system, uav_id, state.drones[uav_id].energy_j)
        flight_j = uav.flight_power_w * pipeline_latency_s
        execution_compute_j = inference_power * exec_layers.get(uav_id, 0) * per_layer_latency
        overlap_compute_j = inference_power * protection_layers.get(uav_id, 0) * per_layer_latency
        tx_j = tx_energy_for_bytes_j(system, uav_id, snapshot_tx.get(uav_id, 0))
        activation_j = (
            tx_energy_for_bytes_j(system, uav_id, system.model.activation_bytes)
            if uav_id in activation_forward_sources
            else 0.0
        )
        energy_per_token = flight_j + execution_compute_j + overlap_compute_j + tx_j + activation_j
        per_token_latency = pipeline_latency_s
        token_rate = 0.0 if per_token_latency <= 0.0 else 1.0 / per_token_latency
        denom = per_token_latency if per_token_latency > 0.0 else 1.0
        out[uav_id] = FutureTokenDiagnostics(
            uav_id=uav_id,
            remaining_energy_j=state.drones[uav_id].energy_j,
            exec_layers=exec_layers.get(uav_id, 0),
            overlap_layers=protection_layers.get(uav_id, 0),
            inference_power_w=inference_power,
            tx_power_w=uav.tx_power_w,
            overlap_power_w=overlap_compute_j / denom,
            snapshot_boundary_power_w=tx_j / denom,
            total_future_power_w=energy_per_token / denom,
            per_token_latency_s=per_token_latency,
            token_rate=token_rate,
            execution_compute_energy_j=execution_compute_j,
            overlap_compute_energy_j=overlap_compute_j,
            flight_energy_j=flight_j,
            snapshot_boundary_energy_j=tx_j,
            activation_forward_energy_j=activation_j,
            energy_per_token_j=energy_per_token,
            predicted_remaining_tokens_i=_remaining_token_capacity(state.drones[uav_id].energy_j, energy_per_token),
        )
    return out


def _energy_per_next_token_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    state: RuntimeState,
    pipeline_latency_s: float,
    recovered_exec_layers_by_uav: dict[int, int],
) -> dict[int, float]:
    diagnostics = _future_token_diagnostics_by_uav(
        system, plan, state, pipeline_latency_s, recovered_exec_layers_by_uav
    )
    return {uav_id: row.energy_per_token_j for uav_id, row in diagnostics.items()}


def _remaining_token_capacity(energy_j: float, energy_per_token_j: float) -> float:
    if energy_j <= 0.0:
        return 0.0
    if energy_per_token_j <= 0.0:
        return float("inf")
    return energy_j / energy_per_token_j


def _expected_remaining_tokens_by_uav(
    state: RuntimeState,
    energy_per_token_by_uav: dict[int, float],
) -> dict[int, float]:
    return {
        uav_id: _remaining_token_capacity(drone.energy_j, energy_per_token_by_uav.get(uav_id, 0.0))
        for uav_id, drone in state.drones.items()
        if drone.is_alive()
    }


def _system_expected_remaining_tokens(expected_by_uav: dict[int, float]) -> float:
    return min(expected_by_uav.values()) if expected_by_uav else 0.0


# ---------------------------------------------------------------------------
# P2/P1-new integration
# ---------------------------------------------------------------------------


def _apply_aerokv_p2_p1_new(
    system: SystemSpec,
    state: RuntimeState,
    recovery: RecoveryResult,
    *,
    beam_width: int = 256,
) -> tuple[bool, str | None, float, float, dict[int, int], set[int], bool, bool | None]:
    """Apply AeroKV P2 and P1-new after a successful recovery.

    Returns ``(valid, reason, reconfig_latency, reconfig_energy,
    recovered_exec_layers_by_uav, failed_sources)``.  Reconfiguration latency and
    energy is simplified as a fixed 200 J overhead; planner latency remains
    zero. The important state transition is the new layout/ring/protection plan.
    """

    print(
        f"[AeroKV][reconfiguration start] token={state.token} "
        f"failed_uav={recovery.failure.failed_uav} alive_uavs={tuple(sorted(_alive_set(state)))}"
    )
    print(
        f"[AeroKV][recovery evidence] live_owner={recovery.live_owner_uav} "
        f"snapshot_owner={recovery.snapshot_owner_uav} live_head_layers={recovery.live_head_layers} "
        f"snapshot_tail_layers={recovery.snapshot_tail_layers} "
        f"latest_snapshot_token={recovery.latest_snapshot_token} "
        f"replay_tokens={recovery.replay_tokens} recovery_latency_s={recovery.recovery_latency_s:.6f}"
    )
    p2 = solve_p2_reconfiguration(
        system,
        state.protection_plan,
        state.protection_runtime,
        token=state.token,
        alive_uavs=_alive_set(state),
        recovered_intervals_by_uav=recovery.recovered_intervals_by_uav,
        energy_by_uav_j={u: state.drones[u].energy_j for u in _alive_set(state)},
    )
    if not p2.valid or p2.layout is None:
        print(f"[AeroKV][P2 failed] reason={p2.invalid_reason or 'p2_reconfiguration_failed'}")
        return False, p2.invalid_reason or "p2_reconfiguration_failed", 0.0, 0.0, {}, set(), False, None

    p1_new = solve_p1_new(system, p2, beam_width=beam_width)
    if not p1_new.valid or p1_new.plan is None or p1_new.surviving_ring is None:
        print(f"[AeroKV][P1-new failed] reason={p1_new.invalid_reason or 'p1_new_failed'}")
        return False, p1_new.invalid_reason or "p1_new_failed", 0.0, 0.0, {}, set(), True, False

    state.layout = p2.layout
    state.ring = p1_new.surviving_ring
    state.protection_plan = p1_new.plan
    state.protection_runtime = _fresh_runtime_for_plan_at_token(system, p1_new.plan, state.token)
    state.execution_assignment = {u: (iv,) for u, iv in p2.layout.intervals.items()}

    # Simplified experiment model: account a fixed 200 J reconfiguration overhead.
    reconfig_energy = 200.0
    alive = _alive_set(state)
    if alive:
        share = reconfig_energy / len(alive)
        for u in alive:
            state.drones[u].energy_j -= share
    print(
        f"[AeroKV][reconfiguration success] token={state.token} "
        f"new_ring={state.ring.uav_ids} reconfiguration_energy_j={reconfig_energy:.3f}"
    )
    return True, None, 0.0, reconfig_energy, {}, set(), True, True




def _apply_baseline_ring_rewire_after_failure(
    system: SystemSpec,
    state: RuntimeState,
    current_plan: ProtectionPlan,
    failed_uav: int,
) -> tuple[bool, str | None, ProtectionPlan]:
    """Remove a failed UAV from the baseline logical ring and plan.

    NP/OO/SO do not run P2/P1-new, but their surviving logical ring must still
    skip failed UAVs.  The compact protection runtime requires ring/layout keys
    to match, so we also remove the failed UAV from the plan layout and
    protection dictionaries.  Recovered layers remain tracked separately in
    recovered_exec_layers_by_uav.
    """

    if failed_uav not in current_plan.ring.uav_ids:
        return True, None, current_plan

    remaining = tuple(u for u in current_plan.ring.uav_ids if u != failed_uav and state.drones[u].is_alive())
    if len(remaining) < 2:
        return False, "baseline_rewire_requires_at_least_two_survivors", current_plan

    new_ring = LogicalRing(remaining)
    new_intervals = {u: current_plan.layout.interval(u) for u in remaining if u in current_plan.layout.intervals}
    new_plan = ProtectionPlan(
        method=current_plan.method,
        layout=ExecutionLayout(new_intervals),
        ring=new_ring,
        head_overlap_depth={u: current_plan.head_overlap_depth[u] for u in new_intervals},
        snapshot_period={u: current_plan.snapshot_period[u] for u in new_intervals},
    )
    # Do not call validate_against here: after baseline hard recovery the static
    # layout no longer exactly covers all model layers. Runtime execution of
    # recovered layers is represented separately by recovered_exec_layers_by_uav.
    state.layout = new_plan.layout
    state.ring = new_plan.ring
    state.protection_plan = new_plan
    state.protection_runtime = _fresh_runtime_for_plan_at_token(system, new_plan, state.token)
    state.execution_assignment = {u: (iv,) for u, iv in new_plan.layout.intervals.items()}
    return True, None, new_plan


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
    system_expected_remaining_tokens: float,
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
        system_expected_remaining_tokens=system_expected_remaining_tokens,
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
    if owner_uav not in alive or owner_uav not in plan.layout.intervals:
        return 0, None, None, 0
    source = plan.ring.pred(owner_uav)
    if source not in alive or source not in plan.layout.intervals or effective_snapshot_period(plan, source) is None:
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
    expected_remaining_tokens_by_uav: dict[int, float],
    future_diagnostics_by_uav: dict[int, FutureTokenDiagnostics] | None = None,
) -> list[UAVTraceRow]:
    rows: list[UAVTraceRow] = []
    alive = _alive_set(state)
    exec_layers = _dynamic_exec_layers_by_uav(plan, alive, recovered_exec_layers_by_uav)
    protection_layers = _dynamic_protection_layers_by_uav(plan, alive)

    for uav_id in sorted(state.drones):
        drone = state.drones[uav_id]
        is_alive = drone.is_alive()
        native = plan.layout.interval(uav_id) if uav_id in plan.layout.intervals else None
        snapshot_layers, latest, stale, buf = _snapshot_view_for_holder(
            plan, state.protection_runtime, state.token, uav_id, alive
        )
        future = None if future_diagnostics_by_uav is None else future_diagnostics_by_uav.get(uav_id)
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
                native_layer_start=native.start if is_alive and native is not None else None,
                native_layer_end=native.end if is_alive and native is not None else None,
                exec_layer_start=native.start if is_alive and native is not None else None,
                exec_layer_end=native.end if is_alive and native is not None else None,
                num_exec_layers=exec_layers.get(uav_id, 0),
                live_overlap_layers=protection_layers.get(uav_id, 0),
                snapshot_layers=snapshot_layers,
                latest_snapshot_token=latest,
                snapshot_staleness_tokens=stale,
                activation_buffer_tokens=buf,
                stage_latency_s=stage_latency_by_uav.get(uav_id, 0.0),
                expected_remaining_tokens=expected_remaining_tokens_by_uav.get(uav_id, 0.0),
                recovered_exec_layers=recovered_exec_layers_by_uav.get(uav_id, 0) if is_alive else 0,
                inference_power_w=0.0 if future is None else future.inference_power_w,
                tx_power_w=0.0 if future is None else future.tx_power_w,
                overlap_power_w=0.0 if future is None else future.overlap_power_w,
                snapshot_boundary_power_w=0.0 if future is None else future.snapshot_boundary_power_w,
                total_future_power_w=0.0 if future is None else future.total_future_power_w,
                per_token_latency_s=0.0 if future is None else future.per_token_latency_s,
                token_rate=0.0 if future is None else future.token_rate,
                energy_per_token_j=0.0 if future is None else future.energy_per_token_j,
                predicted_remaining_tokens_i=0.0 if future is None else future.predicted_remaining_tokens_i,
                activation_forward_energy_per_token_j=0.0 if future is None else future.activation_forward_energy_j,
                snapshot_boundary_energy_per_token_j=0.0 if future is None else future.snapshot_boundary_energy_j,
            )
        )
    return rows


def _make_step_log_row(run_id: str, state: RuntimeState, failures_so_far: list[FailureEvent]) -> StepLogRow:
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


def _progress_line(row: StepLogRow, *, method: str | None = None, max_tokens: int | None = None) -> str:
    label = method or row.run_id
    if max_tokens and max_tokens > 0:
        frac = min(1.0, max(0.0, row.token / max_tokens))
        filled = int(round(frac * 24))
        bar = "#" * filled + "-" * (24 - filled)
        prefix = f"[{label}] [{bar}] {100.0 * frac:6.2f}% "
    else:
        prefix = f"[{label}] "
    return (
        prefix
        + f"token={row.token} "
        + f"time_s={row.time_s:.3f} "
        + f"phase={row.phase} "
        + f"alive={row.num_alive_uavs} "
        + f"avg_energy_used_j={row.avg_energy_used_j:.3f} "
        + f"min_energy_j={row.min_energy_j:.3f} "
        + f"failures={row.failure_history}"
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
    enable_p2_p1_new: bool = True,
) -> RecoverySimulationOutput:
    """Run the standard simulation with failures, recovery, and AeroKV P2/P1-new."""

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

    current_plan = plan
    state = RuntimeState(
        token=0,
        time_s=0.0,
        phase="pre_failure",
        system=system,
        layout=current_plan.layout,
        ring=current_plan.ring,
        protection_plan=current_plan,
        protection_runtime=update_protection_runtime_at_completed_token(system, current_plan, 0).runtime,
        drones=_initial_drone_runtime(system),
        failed_uav=None,
        execution_assignment={u: (iv,) for u, iv in current_plan.layout.intervals.items()},
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
    total_reconfiguration_latency_s = 0.0
    total_reconfiguration_energy_j = 0.0
    total_protection_compute_energy_j = 0.0
    p2_valid: bool | None = None
    p1_new_valid: bool | None = None

    initial_latency = protected_pipeline_latency_s(system, current_plan).pipeline_latency_s
    _refresh_dynamic_memory(system, current_plan, state, recovered_exec_layers_by_uav)
    initial_energy_per_token = _energy_per_next_token_by_uav(
        system, current_plan, state, initial_latency, recovered_exec_layers_by_uav
    )
    initial_future_diagnostics = _future_token_diagnostics_by_uav(
        system, current_plan, state, initial_latency, recovered_exec_layers_by_uav
    )
    initial_expected_by_uav = _expected_remaining_tokens_by_uav(state, initial_energy_per_token)
    initial_system_expected = _system_expected_remaining_tokens(initial_expected_by_uav)
    token_trace = [
        _make_token_row(
            run_id, current_plan.method, system, state, initial_latency, failures_so_far, initial_system_expected
        )
    ]
    stage0 = _dynamic_stage_latency_by_uav(system, current_plan, _alive_set(state), recovered_exec_layers_by_uav, {u: state.drones[u].energy_j for u in _alive_set(state)})
    uav_trace = _make_uav_rows(
        run_id,
        current_plan.method,
        system,
        current_plan,
        state,
        stage0,
        recovered_exec_layers_by_uav,
        initial_expected_by_uav,
        initial_future_diagnostics,
    )
    step_log = [_make_step_log_row(run_id, state, failures_so_far)]

    for token in range(1, max_tokens + 1):
        if invalid_reason is not None:
            break

        alive_before_token = _alive_set(state)
        pipeline_latency = _dynamic_pipeline_latency_s(system, current_plan, alive_before_token, recovered_exec_layers_by_uav, {u: state.drones[u].energy_j for u in alive_before_token})
        if pipeline_latency <= 0.0:
            invalid_reason = "no_alive_execution_stage"
            state.phase = "failed"
            break

        state.token = token
        state.time_s += pipeline_latency
        state.protection_runtime = _freeze_failed_sources_runtime(
            system, current_plan, token, state.protection_runtime, failed_sources
        )

        protection_layers_now = _dynamic_protection_layers_by_uav(current_plan, alive_before_token)
        total_protection_compute_energy_j += sum(
            runtime_inference_power_w(system, uav_id, state.drones[uav_id].energy_j) * layers * runtime_per_layer_latency_s(system, uav_id, state.drones[uav_id].energy_j)
            for uav_id, layers in protection_layers_now.items()
            if uav_id in alive_before_token
        )

        snapshot_tx = _snapshot_tx_bytes_for_alive_sources(system, current_plan, token, alive_before_token)
        _apply_token_energy(
            system,
            current_plan,
            state,
            pipeline_latency_s=pipeline_latency,
            recovered_exec_layers_by_uav=recovered_exec_layers_by_uav,
            snapshot_tx_bytes_by_source=snapshot_tx,
        )

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
                current_plan,
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

            if enable_p2_p1_new and _method_is_aerokv(current_plan):
                state.phase = "reconfiguring"
                (
                    ok,
                    reason,
                    reconfig_latency,
                    reconfig_energy,
                    recovered_exec_layers_by_uav,
                    failed_sources,
                    p2_valid,
                    p1_new_valid,
                ) = (
                    _apply_aerokv_p2_p1_new(system, state, result)
                )
                total_reconfiguration_latency_s += reconfig_latency
                total_reconfiguration_energy_j += reconfig_energy
                current_plan = state.protection_plan
                if not ok:
                    invalid_reason = reason or "p2_p1_new_failed"
                    deadline_met_all = False
                    state.phase = "failed"
                    break
                if reconfig_latency > 0:
                    state.time_s += reconfig_latency
                state.phase = "post_failure"
            else:
                ok, reason, current_plan = _apply_baseline_ring_rewire_after_failure(
                    system, state, current_plan, event.failed_uav
                )
                if not ok:
                    invalid_reason = reason or "baseline_ring_rewire_failed"
                    deadline_met_all = False
                    state.phase = "failed"
                    break

        _refresh_dynamic_memory(system, current_plan, state, recovered_exec_layers_by_uav)

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
        stage = _dynamic_stage_latency_by_uav(system, current_plan, alive_after, recovered_exec_layers_by_uav, {u: state.drones[u].energy_j for u in alive_after})
        next_latency = _dynamic_pipeline_latency_s(
            system,
            current_plan,
            alive_after,
            recovered_exec_layers_by_uav,
            {u: state.drones[u].energy_j for u in alive_after},
        )
        energy_per_next_token = _energy_per_next_token_by_uav(
            system, current_plan, state, next_latency, recovered_exec_layers_by_uav
        )
        future_diagnostics = _future_token_diagnostics_by_uav(
            system, current_plan, state, next_latency, recovered_exec_layers_by_uav
        )
        expected_by_uav = _expected_remaining_tokens_by_uav(state, energy_per_next_token)
        system_expected = _system_expected_remaining_tokens(expected_by_uav)
        token_trace.append(
            _make_token_row(run_id, current_plan.method, system, state, next_latency, failures_so_far, system_expected)
        )
        uav_trace.extend(
            _make_uav_rows(
                run_id,
                current_plan.method,
                system,
                current_plan,
                state,
                stage,
                recovered_exec_layers_by_uav,
                expected_by_uav,
                future_diagnostics,
            )
        )
        log_row = _make_step_log_row(run_id, state, failures_so_far)
        step_log.append(log_row)

        if print_progress and (token % progress_interval_tokens == 0 or token == max_tokens):
            end = "\n" if token == max_tokens or state.phase == "failed" else ""
            print("\r" + _progress_line(log_row, method=plan.method, max_tokens=max_tokens), end=end, flush=True)

    alive_drones = [d for d in state.drones.values() if d.is_alive()]
    terminal_min_energy = min((d.energy_j for d in alive_drones), default=0.0)
    completed = state.token == max_tokens and invalid_reason is None
    mission_success = completed and memory_feasible and energy_feasible and deadline_met_all
    if invalid_reason is None and not deadline_met_all:
        invalid_reason = "recovery_deadline_missed"

    first_failure = failures_so_far[0] if failures_so_far else None
    total_tx = sum(d.cumulative_tx_energy_j for d in state.drones.values())
    remaining_latency = _dynamic_pipeline_latency_s(system, current_plan, _alive_set(state), recovered_exec_layers_by_uav, {u: state.drones[u].energy_j for u in _alive_set(state)})
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
        reconfiguration_latency_s=total_reconfiguration_latency_s,
        remaining_completion_time_s=0.0 if completed else max(0, system.model.n_est - state.token) * remaining_latency,
        mission_complete_s=state.time_s if mission_success else None,
        terminal_min_energy_j=terminal_min_energy,
        protection_compute_energy_j=total_protection_compute_energy_j,
        tx_energy_j=total_tx,
        reconfiguration_energy_j=total_reconfiguration_energy_j,
        mission_success=mission_success,
        invalid_reason=invalid_reason,
        num_failures=len(failures_so_far),
        failure_trace=_failure_history_so_far(failures_so_far),
        expected_failures_per_task=expected_failures_per_task,
        total_recovery_latency_s=total_recovery_latency_s,
        final_system_expected_remaining_tokens=token_trace[-1].system_expected_remaining_tokens if token_trace else 0.0,
        p2_valid=p2_valid,
        p1_new_valid=p1_new_valid,
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
    snapshot_period: int | None = 1,
    expected_failures_per_task: float = 2.5,
    failure_seed: int | None = None,
    run_id: str | None = None,
    progress_interval_tokens: int = 20,
    print_progress: bool = False,
    output_dir: str | Path | None = None,
    enable_p2_p1_new: bool = True,
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
        enable_p2_p1_new=enable_p2_p1_new,
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
    result = run_standard_recovery(output_dir=Path("outputs/standard_recovery"), print_progress=True)
    print(f"wrote recovery traces for run_id={result.summary.run_id}")

# Backward-compatible names.  They now use the real engine with an empty failure
# schedule instead of the removed pre-failure-only implementation.
simulate_fixed_plan_prefailure = lambda system, plan, **kwargs: simulate_fixed_plan_with_recovery(
    system,
    plan,
    failure_events=(),
    **kwargs,
)
write_standard_output = write_recovery_output
write_standard_outputs = write_recovery_output
run_standard_pre_failure = simulate_fixed_plan_prefailure
