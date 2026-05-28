"""Pure accounting formulas for the rewritten AeroKV simulator.

Layer 1 contains no planner, no recovery logic, and no simulator loop.  The
functions here are deterministic calculations of memory, latency, compute
energy, flight energy, and TX-only communication energy for a given system,
layout, protection plan, token, and compact protection runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import ExecutionLayout, ProtectionPlan, ProtectionRuntime, SystemSpec


@dataclass(frozen=True)
class MemoryBreakdown:
    """Per-UAV memory components in bytes."""

    native_bytes: float
    live_overlap_bytes: float
    snapshot_bytes: float
    activation_buffer_bytes: float
    full_mirror_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        return (
            self.native_bytes
            + self.live_overlap_bytes
            + self.snapshot_bytes
            + self.activation_buffer_bytes
            + self.full_mirror_bytes
        )


@dataclass(frozen=True)
class EnergyPerTokenBreakdown:
    """Per-UAV steady-state per-token energy components in joules."""

    flight_j: float
    execution_compute_j: float
    protection_compute_j: float
    tx_j: float

    @property
    def compute_j(self) -> float:
        return self.execution_compute_j + self.protection_compute_j

    @property
    def total_j(self) -> float:
        return self.flight_j + self.execution_compute_j + self.protection_compute_j + self.tx_j


@dataclass(frozen=True)
class LatencyBreakdown:
    """Pipeline latency and per-UAV stage latency in seconds."""

    stage_latency_by_uav: Mapping[int, float]
    pipeline_latency_s: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_token(system: SystemSpec, token: int) -> None:
    if not (0 <= token <= system.model.n_est):
        raise ValueError(f"token must be within [0, {system.model.n_est}], got {token}")


def validate_uav_in_plan(plan: ProtectionPlan, uav_id: int) -> None:
    if uav_id not in plan.layout.intervals:
        raise KeyError(f"uav_id {uav_id} is not present in the protection layout")


# ---------------------------------------------------------------------------
# Layer ownership / protection geometry
# ---------------------------------------------------------------------------


def native_layers(plan: ProtectionPlan, uav_id: int) -> int:
    """Number of layers natively executed by ``uav_id`` in the current layout."""

    validate_uav_in_plan(plan, uav_id)
    return plan.layout.interval(uav_id).width


def live_overlap_layers_for_source(plan: ProtectionPlan, source_uav: int) -> int:
    """Leading layers of ``source_uav`` live-overlapped on pred(source)."""

    validate_uav_in_plan(plan, source_uav)
    shard_width = plan.layout.interval(source_uav).width
    return min(max(plan.head_overlap_depth[source_uav], 0), shard_width)


def snapshot_tail_layers_for_source(plan: ProtectionPlan, source_uav: int) -> int:
    """Tail layers of ``source_uav`` protected by Boundary Snapshot on succ(source)."""

    validate_uav_in_plan(plan, source_uav)
    return max(0, plan.layout.interval(source_uav).width - live_overlap_layers_for_source(plan, source_uav))


def live_overlap_source_for_owner(plan: ProtectionPlan, owner_uav: int) -> int:
    """Source whose live-overlap head is stored/computed by ``owner_uav``."""

    return plan.ring.succ(owner_uav)


def snapshot_source_for_owner(plan: ProtectionPlan, owner_uav: int) -> int:
    """Source whose snapshot tail is stored by ``owner_uav``."""

    return plan.ring.pred(owner_uav)


def live_overlap_layers_stored_on(plan: ProtectionPlan, owner_uav: int) -> int:
    source = live_overlap_source_for_owner(plan, owner_uav)
    return live_overlap_layers_for_source(plan, source)


def snapshot_layers_stored_on(plan: ProtectionPlan, owner_uav: int) -> int:
    source = snapshot_source_for_owner(plan, owner_uav)
    if plan.snapshot_period[source] is None:
        return 0
    return snapshot_tail_layers_for_source(plan, source)


# ---------------------------------------------------------------------------
# Snapshot freshness / bytes
# ---------------------------------------------------------------------------


def snapshot_token_at_or_before(token: int, period: int | None) -> int | None:
    """Latest completed token included in a periodic snapshot."""

    if period is None:
        return None
    if period <= 0:
        raise ValueError("snapshot period must be positive or None")
    if token < 0:
        raise ValueError("token must be non-negative")
    return (token // period) * period


def snapshot_due_at_completed_token(token: int, period: int | None) -> bool:
    """Whether a periodic snapshot is due after completing ``token`` tokens."""

    return period is not None and token > 0 and token % period == 0


def snapshot_staleness_tokens(token: int, latest_snapshot_token: int | None) -> int | None:
    if latest_snapshot_token is None:
        return None
    if latest_snapshot_token > token:
        raise ValueError("latest_snapshot_token cannot exceed token")
    return token - latest_snapshot_token


def activation_buffer_tokens_for_source(
    plan: ProtectionPlan,
    source_uav: int,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> int:
    """Tokens buffered since the latest snapshot for a source's snapshot tail."""

    period = plan.snapshot_period[source_uav]
    if period is None or snapshot_tail_layers_for_source(plan, source_uav) == 0:
        return 0
    if runtime is not None and source_uav in runtime.activation_buffer_tokens:
        return runtime.activation_buffer_tokens[source_uav]
    latest = snapshot_token_at_or_before(token, period)
    assert latest is not None
    return token - latest


def latest_snapshot_token_for_source(
    plan: ProtectionPlan,
    source_uav: int,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> int | None:
    """Latest snapshot token for ``source_uav`` using runtime state if available."""

    period = plan.snapshot_period[source_uav]
    if period is None or snapshot_tail_layers_for_source(plan, source_uav) == 0:
        return None
    if runtime is not None and source_uav in runtime.latest_snapshot_token:
        latest = runtime.latest_snapshot_token[source_uav]
        if latest > token:
            raise ValueError("runtime latest snapshot token cannot exceed current token")
        return latest
    return snapshot_token_at_or_before(token, period)


def snapshot_tx_bytes_at_completed_token(
    system: SystemSpec,
    plan: ProtectionPlan,
    source_uav: int,
    token: int,
) -> int:
    """Exact Boundary Snapshot payload emitted by ``source_uav`` at a completed token.

    This is zero on non-snapshot tokens.  At token ``T`` divisible by the period,
    the payload is ``tail_layers * period * kv_bytes_per_token_layer``.
    """

    validate_token(system, token)
    period = plan.snapshot_period[source_uav]
    if period is None or not snapshot_due_at_completed_token(token, period):
        return 0
    tail_layers = snapshot_tail_layers_for_source(plan, source_uav)
    return int(tail_layers * period * system.model.kv_bytes_per_token_layer)


def average_snapshot_tx_bytes_per_token(
    system: SystemSpec,
    plan: ProtectionPlan,
    source_uav: int,
) -> float:
    """Average per-token TX payload for the source's snapshot stream.

    For periodic Boundary Snapshot, this equals one token of KV for each tail
    layer.  For the ideal full-mirror reference, this mirrors the old
    non-deployable reference model: repeatedly transmit the growing native KV,
    whose generation-average payload is width * kv * n_est / 2 per token.
    """

    if plan.full_mirror:
        width = plan.layout.interval(source_uav).width
        return width * system.model.kv_bytes_per_token_layer * system.model.n_est / 2.0
    if plan.snapshot_period[source_uav] is None:
        return 0.0
    return snapshot_tail_layers_for_source(plan, source_uav) * system.model.kv_bytes_per_token_layer


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def memory_breakdown_bytes(
    system: SystemSpec,
    plan: ProtectionPlan,
    uav_id: int,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> MemoryBreakdown:
    """Compute per-component memory for one UAV at a completed token.

    Native and live-overlap memory include weights and KV.  Snapshot memory
    stores KV only.  The boundary activation buffer stores activations since the
    latest snapshot.  Full mirror memory, when enabled, stores the successor's
    native KV only, matching the old reference convention.
    """

    validate_token(system, token)
    validate_uav_in_plan(plan, uav_id)
    model = system.model

    native = native_layers(plan, uav_id) * (model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer)

    overlap_layers = live_overlap_layers_stored_on(plan, uav_id)
    live_overlap = overlap_layers * (model.weight_bytes_per_layer + token * model.kv_bytes_per_token_layer)

    snapshot_source = snapshot_source_for_owner(plan, uav_id)
    latest = latest_snapshot_token_for_source(plan, snapshot_source, token, runtime)
    snapshot_layers = snapshot_layers_stored_on(plan, uav_id)
    snapshot = 0.0 if latest is None else snapshot_layers * latest * model.kv_bytes_per_token_layer

    activation_tokens = activation_buffer_tokens_for_source(plan, snapshot_source, token, runtime)
    activation_buffer = activation_tokens * model.activation_bytes if snapshot_layers > 0 else 0.0

    full_mirror = 0.0
    if plan.full_mirror:
        mirror_source = plan.ring.succ(uav_id)
        full_mirror = plan.layout.interval(mirror_source).width * token * model.kv_bytes_per_token_layer

    return MemoryBreakdown(
        native_bytes=float(native),
        live_overlap_bytes=float(live_overlap),
        snapshot_bytes=float(snapshot),
        activation_buffer_bytes=float(activation_buffer),
        full_mirror_bytes=float(full_mirror),
    )


def memory_by_uav_bytes(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> dict[int, float]:
    return {
        uav_id: memory_breakdown_bytes(system, plan, uav_id, token, runtime).total_bytes
        for uav_id in plan.layout.intervals
    }


def memory_feasible(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> bool:
    usage = memory_by_uav_bytes(system, plan, token, runtime)
    return all(usage[uav_id] <= system.uav(uav_id).memory_budget_bytes for uav_id in usage)


# ---------------------------------------------------------------------------
# Latency accounting
# ---------------------------------------------------------------------------


def execution_stage_latency_s(system: SystemSpec, layout: ExecutionLayout, uav_id: int) -> float:
    layers = layout.interval(uav_id).width
    return layers * system.uav(uav_id).per_layer_latency_s


def protected_stage_latency_s(system: SystemSpec, plan: ProtectionPlan, uav_id: int) -> float:
    layers = native_layers(plan, uav_id) + live_overlap_layers_stored_on(plan, uav_id)
    return layers * system.uav(uav_id).per_layer_latency_s


def pipeline_latency_s(
    system: SystemSpec,
    layout: ExecutionLayout,
    active_uavs: tuple[int, ...] | None = None,
) -> LatencyBreakdown:
    if active_uavs is None:
        active_uavs = layout.executing_uavs()
    stage = {uav_id: execution_stage_latency_s(system, layout, uav_id) for uav_id in active_uavs}
    return LatencyBreakdown(stage_latency_by_uav=stage, pipeline_latency_s=max(stage.values()) if stage else 0.0)


def protected_pipeline_latency_s(system: SystemSpec, plan: ProtectionPlan) -> LatencyBreakdown:
    active_uavs = plan.layout.executing_uavs()
    stage = {uav_id: protected_stage_latency_s(system, plan, uav_id) for uav_id in active_uavs}
    return LatencyBreakdown(stage_latency_by_uav=stage, pipeline_latency_s=max(stage.values()) if stage else 0.0)


# ---------------------------------------------------------------------------
# Energy accounting
# ---------------------------------------------------------------------------


def compute_energy_for_layers_per_token_j(system: SystemSpec, uav_id: int, layers: int) -> float:
    if layers < 0:
        raise ValueError("layers must be non-negative")
    uav = system.uav(uav_id)
    return uav.inference_power_w * layers * uav.per_layer_latency_s


def execution_compute_energy_per_token_j(system: SystemSpec, layout: ExecutionLayout, uav_id: int) -> float:
    return compute_energy_for_layers_per_token_j(system, uav_id, layout.interval(uav_id).width)


def protection_compute_energy_per_token_j(system: SystemSpec, plan: ProtectionPlan, uav_id: int) -> float:
    return compute_energy_for_layers_per_token_j(system, uav_id, live_overlap_layers_stored_on(plan, uav_id))


def protected_compute_energy_per_token_j(system: SystemSpec, plan: ProtectionPlan, uav_id: int) -> float:
    return execution_compute_energy_per_token_j(system, plan.layout, uav_id) + protection_compute_energy_per_token_j(
        system, plan, uav_id
    )


def flight_energy_for_duration_j(system: SystemSpec, uav_id: int, duration_s: float) -> float:
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    return system.uav(uav_id).flight_power_w * duration_s


def tx_energy_for_bytes_j(system: SystemSpec, source_uav: int, num_bytes: float) -> float:
    """TX-only communication energy for bytes sent by ``source_uav``."""

    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    uav = system.uav(source_uav)
    return uav.tx_power_w * num_bytes * 8.0 / uav.link_bps


def average_snapshot_tx_energy_per_token_j(system: SystemSpec, plan: ProtectionPlan, source_uav: int) -> float:
    return tx_energy_for_bytes_j(system, source_uav, average_snapshot_tx_bytes_per_token(system, plan, source_uav))


def snapshot_tx_energy_at_completed_token_j(
    system: SystemSpec,
    plan: ProtectionPlan,
    source_uav: int,
    token: int,
) -> float:
    return tx_energy_for_bytes_j(system, source_uav, snapshot_tx_bytes_at_completed_token(system, plan, source_uav, token))


def steady_state_energy_per_token_breakdown(
    system: SystemSpec,
    plan: ProtectionPlan,
    uav_id: int,
    pipeline_latency: float | None = None,
) -> EnergyPerTokenBreakdown:
    """Average steady-state per-token energy for one UAV under a protection plan."""

    if pipeline_latency is None:
        pipeline_latency = protected_pipeline_latency_s(system, plan).pipeline_latency_s
    return EnergyPerTokenBreakdown(
        flight_j=flight_energy_for_duration_j(system, uav_id, pipeline_latency),
        execution_compute_j=execution_compute_energy_per_token_j(system, plan.layout, uav_id),
        protection_compute_j=protection_compute_energy_per_token_j(system, plan, uav_id),
        tx_j=average_snapshot_tx_energy_per_token_j(system, plan, uav_id),
    )


def steady_state_energy_per_token_by_uav_j(system: SystemSpec, plan: ProtectionPlan) -> dict[int, EnergyPerTokenBreakdown]:
    latency = protected_pipeline_latency_s(system, plan).pipeline_latency_s
    return {
        uav_id: steady_state_energy_per_token_breakdown(system, plan, uav_id, latency)
        for uav_id in plan.layout.intervals
    }
