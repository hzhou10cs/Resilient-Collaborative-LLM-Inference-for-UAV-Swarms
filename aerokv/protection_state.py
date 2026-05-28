"""Compact per-token protection runtime updates for AeroKV.

Layer 2 still contains no simulator loop, no failure recovery, no P1/P2, no event
log, and no KV segment debug table.  It only updates the compact protection
state needed by later layers:

- latest_snapshot_token[source_uav]
- activation_buffer_tokens[source_uav]

Token convention: ``token`` means the number of completed generated tokens.  For
example, at ``token == 4`` and period 4, a snapshot has just been completed and
``latest_snapshot_token == 4``.

Direction convention:

- From source UAV i: i's head overlap is stored/computed on pred(i), and i's
  tail snapshot is stored on succ(i).
- From holder UAV i: i stores/computes succ(i)'s head overlap, and i stores
  pred(i)'s tail snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .accounting import (
    activation_buffer_tokens_for_source,
    effective_snapshot_period,
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    live_overlap_layers_stored_on,
    live_overlap_source_for_owner,
    snapshot_due_at_completed_token,
    snapshot_layers_stored_on,
    snapshot_source_for_owner,
    snapshot_staleness_tokens,
    snapshot_tail_layers_for_source,
    snapshot_token_at_or_before,
    snapshot_tx_bytes_at_completed_token,
    validate_token,
)
from .specs import ProtectionPlan, ProtectionRuntime, SystemSpec


@dataclass(frozen=True)
class ProtectionRuntimeUpdate:
    """Result of computing protection runtime state at one completed token."""

    token: int
    runtime: ProtectionRuntime
    snapshot_tx_bytes_by_source: Mapping[int, int]
    total_snapshot_tx_bytes: int
    snapshot_sources: tuple[int, ...]


@dataclass(frozen=True)
class SourceProtectionView:
    """Source-side view for one UAV's protected shard at one completed token."""

    source_uav: int
    live_overlap_owner_uav: int
    snapshot_owner_uav: int
    head_overlap_layers: int
    snapshot_tail_layers: int
    snapshot_period: int | None
    latest_snapshot_token: int | None
    snapshot_staleness_tokens: int | None
    activation_buffer_tokens: int
    snapshot_due: bool
    snapshot_tx_bytes: int


@dataclass(frozen=True)
class OwnerProtectionView:
    """Holder-side view for one UAV at one completed token.

    For owner UAV i, the live-overlap source is succ(i), and the snapshot source
    is pred(i).
    """

    owner_uav: int
    live_overlap_source_uav: int
    live_overlap_layers: int
    snapshot_source_uav: int
    snapshot_layers: int
    latest_snapshot_token: int | None
    snapshot_staleness_tokens: int | None
    activation_buffer_tokens: int


# ---------------------------------------------------------------------------
# Plan / source helpers
# ---------------------------------------------------------------------------


def validate_plan_ring_matches_layout(plan: ProtectionPlan) -> None:
    """Require the logical ring and execution layout to contain the same UAV ids."""

    ring_ids = set(plan.ring.uav_ids)
    layout_ids = set(plan.layout.intervals.keys())
    if ring_ids != layout_ids:
        raise ValueError(f"ring/layout UAV mismatch: ring={sorted(ring_ids)}, layout={sorted(layout_ids)}")



def plan_sources(plan: ProtectionPlan) -> tuple[int, ...]:
    """Source UAV ids in logical-ring order."""

    validate_plan_ring_matches_layout(plan)
    return plan.ring.uav_ids



def source_has_snapshot_tail(plan: ProtectionPlan, source_uav: int) -> bool:
    """Whether the source has a non-empty tail protected by Boundary Snapshot."""

    return effective_snapshot_period(plan, source_uav) is not None and snapshot_tail_layers_for_source(plan, source_uav) > 0


# ---------------------------------------------------------------------------
# Runtime update
# ---------------------------------------------------------------------------


def runtime_after_completed_token(system: SystemSpec, plan: ProtectionPlan, token: int) -> ProtectionRuntime:
    """Return compact protection runtime state after completing ``token`` tokens.

    The function is deterministic and stateless: it computes the correct runtime
    state directly from the completed token index and the protection plan.
    """

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)

    latest_snapshot_token: dict[int, int] = {}
    activation_buffer_tokens: dict[int, int] = {}

    for source_uav in plan_sources(plan):
        if not source_has_snapshot_tail(plan, source_uav):
            continue
        period = effective_snapshot_period(plan, source_uav)
        latest = snapshot_token_at_or_before(token, period)
        assert latest is not None
        latest_snapshot_token[source_uav] = latest
        activation_buffer_tokens[source_uav] = token - latest

    runtime = ProtectionRuntime(
        latest_snapshot_token=latest_snapshot_token,
        activation_buffer_tokens=activation_buffer_tokens,
    )
    validate_runtime_consistency(system, plan, token, runtime)
    return runtime



def update_protection_runtime_at_completed_token(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
) -> ProtectionRuntimeUpdate:
    """Compute runtime state and exact snapshot TX payloads at one completed token."""

    runtime = runtime_after_completed_token(system, plan, token)
    snapshot_tx_bytes_by_source = {
        source_uav: snapshot_tx_bytes_at_completed_token(system, plan, source_uav, token)
        for source_uav in plan_sources(plan)
    }
    snapshot_sources = tuple(
        source_uav for source_uav, num_bytes in snapshot_tx_bytes_by_source.items() if num_bytes > 0
    )
    return ProtectionRuntimeUpdate(
        token=token,
        runtime=runtime,
        snapshot_tx_bytes_by_source=snapshot_tx_bytes_by_source,
        total_snapshot_tx_bytes=sum(snapshot_tx_bytes_by_source.values()),
        snapshot_sources=snapshot_sources,
    )



def validate_runtime_consistency(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    runtime: ProtectionRuntime,
) -> None:
    """Validate the compact runtime state for one completed token."""

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)

    for source_uav in plan_sources(plan):
        active = source_has_snapshot_tail(plan, source_uav)
        has_latest = source_uav in runtime.latest_snapshot_token
        has_buffer = source_uav in runtime.activation_buffer_tokens

        if not active:
            if has_latest and runtime.latest_snapshot_token[source_uav] != 0:
                raise ValueError(f"inactive snapshot source {source_uav} has nonzero latest snapshot token")
            if has_buffer and runtime.activation_buffer_tokens[source_uav] != 0:
                raise ValueError(f"inactive snapshot source {source_uav} has nonzero activation buffer")
            continue

        if not has_latest:
            raise ValueError(f"active snapshot source {source_uav} is missing latest_snapshot_token")
        if not has_buffer:
            raise ValueError(f"active snapshot source {source_uav} is missing activation_buffer_tokens")

        latest = runtime.latest_snapshot_token[source_uav]
        buffered = runtime.activation_buffer_tokens[source_uav]
        period = effective_snapshot_period(plan, source_uav)
        assert period is not None

        if latest < 0 or latest > token:
            raise ValueError(f"invalid latest snapshot token {latest} for source {source_uav}")
        if buffered < 0:
            raise ValueError(f"negative activation buffer for source {source_uav}")
        if latest + buffered != token:
            raise ValueError(
                f"runtime mismatch for source {source_uav}: latest {latest} + buffer {buffered} != token {token}"
            )
        if latest % period != 0:
            raise ValueError(f"latest snapshot token {latest} is not aligned to period {period}")
        if buffered >= period:
            raise ValueError(f"activation buffer {buffered} must be less than period {period}")


# ---------------------------------------------------------------------------
# Views for later trace generation
# ---------------------------------------------------------------------------


def source_protection_view(
    system: SystemSpec,
    plan: ProtectionPlan,
    source_uav: int,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> SourceProtectionView:
    """Return source-side protection metadata for trace construction."""

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)
    if runtime is None:
        runtime = runtime_after_completed_token(system, plan, token)

    latest = latest_snapshot_token_for_source(plan, source_uav, token, runtime)
    staleness = snapshot_staleness_tokens(token, latest)
    period = effective_snapshot_period(plan, source_uav)
    tail_layers = snapshot_tail_layers_for_source(plan, source_uav)
    active_snapshot = source_has_snapshot_tail(plan, source_uav)
    due = active_snapshot and snapshot_due_at_completed_token(token, period)

    return SourceProtectionView(
        source_uav=source_uav,
        live_overlap_owner_uav=plan.ring.pred(source_uav),
        snapshot_owner_uav=plan.ring.succ(source_uav),
        head_overlap_layers=live_overlap_layers_for_source(plan, source_uav),
        snapshot_tail_layers=tail_layers if active_snapshot else 0,
        snapshot_period=period if active_snapshot else None,
        latest_snapshot_token=latest,
        snapshot_staleness_tokens=staleness,
        activation_buffer_tokens=activation_buffer_tokens_for_source(plan, source_uav, token, runtime),
        snapshot_due=due,
        snapshot_tx_bytes=snapshot_tx_bytes_at_completed_token(system, plan, source_uav, token),
    )



def owner_protection_view(
    system: SystemSpec,
    plan: ProtectionPlan,
    owner_uav: int,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> OwnerProtectionView:
    """Return holder-side protection metadata for one UAV.

    This is the view that should feed ``UAVTraceRow`` later.
    """

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)
    if runtime is None:
        runtime = runtime_after_completed_token(system, plan, token)

    live_source = live_overlap_source_for_owner(plan, owner_uav)
    snapshot_source = snapshot_source_for_owner(plan, owner_uav)
    latest = latest_snapshot_token_for_source(plan, snapshot_source, token, runtime)

    return OwnerProtectionView(
        owner_uav=owner_uav,
        live_overlap_source_uav=live_source,
        live_overlap_layers=live_overlap_layers_stored_on(plan, owner_uav),
        snapshot_source_uav=snapshot_source,
        snapshot_layers=snapshot_layers_stored_on(plan, owner_uav),
        latest_snapshot_token=latest,
        snapshot_staleness_tokens=snapshot_staleness_tokens(token, latest),
        activation_buffer_tokens=activation_buffer_tokens_for_source(plan, snapshot_source, token, runtime),
    )



def owner_views_by_uav(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    runtime: ProtectionRuntime | None = None,
) -> dict[int, OwnerProtectionView]:
    """Return holder-side views for all UAVs in the logical ring."""

    if runtime is None:
        runtime = runtime_after_completed_token(system, plan, token)
    return {
        owner_uav: owner_protection_view(system, plan, owner_uav, token, runtime)
        for owner_uav in plan_sources(plan)
    }


def advance_protection_runtime_for_completed_token(
    system: SystemSpec,
    plan: ProtectionPlan,
    previous_runtime: ProtectionRuntime,
    token: int,
    alive_uavs: tuple[int, ...],
) -> ProtectionRuntimeUpdate:
    """Advance compact protection state for a live run with failed UAVs.

    Unlike ``runtime_after_completed_token``, this function is stateful and
    respects failed UAVs.  A source emits a new snapshot only if both the source
    and its snapshot holder succ(source) are alive.  Sources or holders that are
    no longer alive are removed from the active compact runtime.
    """

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)
    alive = set(alive_uavs)

    latest_snapshot_token: dict[int, int] = {}
    activation_buffer_tokens: dict[int, int] = {}
    snapshot_tx_bytes_by_source: dict[int, int] = {}

    for source_uav in plan_sources(plan):
        snapshot_tx_bytes_by_source[source_uav] = 0
        if not source_has_snapshot_tail(plan, source_uav):
            continue
        holder = plan.ring.succ(source_uav)
        if source_uav not in alive or holder not in alive:
            continue

        period = effective_snapshot_period(plan, source_uav)
        assert period is not None

        prev_latest = previous_runtime.latest_snapshot_token.get(source_uav)
        if prev_latest is None:
            prev_latest = snapshot_token_at_or_before(max(0, token - 1), period)
            assert prev_latest is not None

        if snapshot_due_at_completed_token(token, period):
            latest = token
            buffered = 0
            snapshot_tx_bytes_by_source[source_uav] = snapshot_tx_bytes_at_completed_token(
                system,
                plan,
                source_uav,
                token,
            )
        else:
            latest = prev_latest
            buffered = token - latest

        latest_snapshot_token[source_uav] = latest
        activation_buffer_tokens[source_uav] = buffered

    runtime = ProtectionRuntime(
        latest_snapshot_token=latest_snapshot_token,
        activation_buffer_tokens=activation_buffer_tokens,
    )
    validate_runtime_consistency_for_alive_sources(system, plan, token, runtime, alive_uavs)
    snapshot_sources = tuple(
        source_uav for source_uav, num_bytes in snapshot_tx_bytes_by_source.items() if num_bytes > 0
    )
    return ProtectionRuntimeUpdate(
        token=token,
        runtime=runtime,
        snapshot_tx_bytes_by_source=snapshot_tx_bytes_by_source,
        total_snapshot_tx_bytes=sum(snapshot_tx_bytes_by_source.values()),
        snapshot_sources=snapshot_sources,
    )


def validate_runtime_consistency_for_alive_sources(
    system: SystemSpec,
    plan: ProtectionPlan,
    token: int,
    runtime: ProtectionRuntime,
    alive_uavs: tuple[int, ...],
) -> None:
    """Validate runtime after some UAVs have failed.

    Only sources whose source and snapshot holder are both alive are required to
    have compact snapshot state.
    """

    validate_token(system, token)
    validate_plan_ring_matches_layout(plan)
    alive = set(alive_uavs)
    for source_uav in plan_sources(plan):
        holder = plan.ring.succ(source_uav)
        active = source_has_snapshot_tail(plan, source_uav) and source_uav in alive and holder in alive
        has_latest = source_uav in runtime.latest_snapshot_token
        has_buffer = source_uav in runtime.activation_buffer_tokens
        if not active:
            if has_latest or has_buffer:
                raise ValueError(f"inactive source {source_uav} unexpectedly has compact runtime state")
            continue
        if not has_latest or not has_buffer:
            raise ValueError(f"active source {source_uav} missing compact runtime state")
        latest = runtime.latest_snapshot_token[source_uav]
        buffered = runtime.activation_buffer_tokens[source_uav]
        period = effective_snapshot_period(plan, source_uav)
        assert period is not None
        if latest < 0 or latest > token:
            raise ValueError(f"invalid latest snapshot token {latest} for source {source_uav}")
        if buffered < 0 or latest + buffered != token:
            raise ValueError(
                f"runtime mismatch for source {source_uav}: latest {latest} + buffer {buffered} != token {token}"
            )
        if latest % period != 0:
            raise ValueError(f"latest snapshot token {latest} is not aligned to period {period}")
        if buffered >= period:
            raise ValueError(f"activation buffer {buffered} must be less than period {period}")
