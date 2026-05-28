"""State-constrained P2 reconfiguration.

This module deliberately implements correctness before aggressive optimization.
It never falls back to an unconstrained uniform repartition.  A surviving UAV may
claim a layer only if that layer is available to it from one of the compact
AeroKV state sources:

- its own native execution shard;
- a live-overlapped head shard that it holds for its ring successor;
- a snapshot-recoverable tail shard that it holds for its ring predecessor;
- a layer interval already recovered onto the UAV by the recovery step.

The solver returns a contiguous exact-cover layout over surviving UAVs.  Each
active surviving UAV must receive at least one layer; zero-width assignments are
rejected because they create invalid ring/protection semantics after P1-new.
It uses a small dynamic program over layer boundaries and chooses the feasible
layout with the minimum maximum stage latency.  Reconfiguration data movement is
not modeled here; illegal movement is rejected rather than hidden behind a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..accounting import (
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    snapshot_tail_layers_for_source,
)
from ..specs import ExecutionLayout, LayerInterval, ProtectionPlan, ProtectionRuntime, SystemSpec


@dataclass(frozen=True)
class AvailabilityIndex:
    """Layer availability sets for surviving UAVs."""

    available_layers_by_uav: Mapping[int, frozenset[int]]

    def can_claim(self, uav_id: int, interval: LayerInterval) -> bool:
        layers = self.available_layers_by_uav.get(uav_id, frozenset())
        return all(layer in layers for layer in range(interval.start, interval.end))


@dataclass(frozen=True)
class P2Result:
    valid: bool
    invalid_reason: str | None
    active_uavs: tuple[int, ...]
    layout: ExecutionLayout | None
    max_stage_latency_s: float
    stage_latency_by_uav: Mapping[int, float]
    availability: AvailabilityIndex


def _layers(interval: LayerInterval) -> set[int]:
    return set(range(interval.start, interval.end))


def _head_interval(plan: ProtectionPlan, source_uav: int) -> LayerInterval:
    shard = plan.layout.interval(source_uav)
    k = live_overlap_layers_for_source(plan, source_uav)
    return LayerInterval(shard.start, shard.start + k)


def _tail_interval(plan: ProtectionPlan, source_uav: int) -> LayerInterval:
    shard = plan.layout.interval(source_uav)
    k = live_overlap_layers_for_source(plan, source_uav)
    return LayerInterval(shard.start + k, shard.end)


def build_availability_index(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    *,
    token: int,
    alive_uavs: set[int],
    recovered_intervals_by_uav: Mapping[int, tuple[LayerInterval, ...]] | None = None,
) -> AvailabilityIndex:
    """Build the state-constrained layer availability index.

    Holder semantics are used:
      - holder u stores/computes succ(u)'s live-overlap head;
      - holder u stores pred(u)'s boundary snapshot tail.
    """

    if recovered_intervals_by_uav is None:
        recovered_intervals_by_uav = {}

    out: dict[int, set[int]] = {u: set() for u in alive_uavs}
    for uav_id in alive_uavs:
        # Native shard of the surviving UAV.
        out[uav_id].update(_layers(plan.layout.interval(uav_id)))

        # Live-overlap head held for successor.
        succ = plan.ring.succ(uav_id)
        if succ in plan.layout.intervals:
            out[uav_id].update(_layers(_head_interval(plan, succ)))

        # Snapshot tail held for predecessor.  It is claimable only if there is
        # an actual snapshot for that source in the runtime state.
        pred = plan.ring.pred(uav_id)
        if pred in plan.layout.intervals and plan.snapshot_period[pred] is not None:
            tail = _tail_interval(plan, pred)
            latest = latest_snapshot_token_for_source(plan, pred, token, runtime)
            if latest is not None and latest <= token and tail.width > 0:
                out[uav_id].update(_layers(tail))

        # Intervals already materialized by the recovery step.
        for interval in recovered_intervals_by_uav.get(uav_id, ()):
            out[uav_id].update(_layers(interval))

    return AvailabilityIndex({u: frozenset(v) for u, v in out.items()})


def _active_uavs_in_layer_order(plan: ProtectionPlan, alive_uavs: set[int]) -> tuple[int, ...]:
    return tuple(
        uav_id
        for uav_id, _ in sorted(plan.layout.intervals.items(), key=lambda item: item[1].start)
        if uav_id in alive_uavs
    )


def _memory_ok_for_assignment(
    system: SystemSpec,
    uav_id: int,
    interval: LayerInterval,
    token: int,
) -> bool:
    # P2 correctness check for the new execution shard.  Protection rebuilding
    # is handled by P1-new, so this check does not add new overlap/snapshot state.
    required = interval.width * (system.model.weight_bytes_per_layer + token * system.model.kv_bytes_per_token_layer)
    return required <= system.uav(uav_id).memory_budget_bytes


def solve_p2_reconfiguration(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    *,
    token: int,
    alive_uavs: set[int],
    recovered_intervals_by_uav: Mapping[int, tuple[LayerInterval, ...]] | None = None,
) -> P2Result:
    """Solve a state-constrained contiguous P2 layout.

    Objective: minimize max per-token stage latency among surviving UAVs.
    Failure: returns valid=False; it does not emit an unconstrained fallback.
    """

    if not alive_uavs:
        availability = AvailabilityIndex({})
        return P2Result(False, "no_alive_uavs", tuple(), None, 0.0, {}, availability)

    availability = build_availability_index(
        system,
        plan,
        runtime,
        token=token,
        alive_uavs=alive_uavs,
        recovered_intervals_by_uav=recovered_intervals_by_uav,
    )
    active = _active_uavs_in_layer_order(plan, alive_uavs)
    if not active:
        return P2Result(False, "no_active_uavs_in_layout", tuple(), None, 0.0, {}, availability)

    num_layers = system.model.num_layers
    if len(active) > num_layers:
        return P2Result(
            False,
            "not_enough_layers_for_min_one_layer_per_active_uav",
            active,
            None,
            0.0,
            {},
            availability,
        )
    inf = float("inf")

    # dp[i][b] is best max-stage-latency after assigning layers [0,b) to the
    # first i active UAVs.  Parent stores the previous boundary a.  Because
    # every active UAV must receive at least one layer, feasible states satisfy
    # b >= i and each transition interval [a,b) has width >= 1.
    dp = [[inf] * (num_layers + 1) for _ in range(len(active) + 1)]
    parent: list[list[int | None]] = [[None] * (num_layers + 1) for _ in range(len(active) + 1)]
    dp[0][0] = 0.0

    for i, uav_id in enumerate(active, start=1):
        per_layer = system.uav(uav_id).per_layer_latency_s
        # At least i layers must have been assigned to the first i active UAVs.
        for b in range(i, num_layers + 1):
            # Previous boundary must leave at least one layer for current UAV.
            # It must also leave at least one layer per previous active UAV.
            for a in range(i - 1, b):
                if dp[i - 1][a] == inf:
                    continue
                interval = LayerInterval(a, b)
                if interval.width < 1:
                    continue
                if not availability.can_claim(uav_id, interval):
                    continue
                if not _memory_ok_for_assignment(system, uav_id, interval, token):
                    continue
                stage = interval.width * per_layer
                value = max(dp[i - 1][a], stage)
                if value < dp[i][b]:
                    dp[i][b] = value
                    parent[i][b] = a

    if dp[len(active)][num_layers] == inf:
        return P2Result(False, "no_state_constrained_exact_cover", active, None, 0.0, {}, availability)

    intervals: dict[int, LayerInterval] = {}
    b = num_layers
    for i in range(len(active), 0, -1):
        a = parent[i][b]
        if a is None:
            return P2Result(False, "missing_dp_parent", active, None, 0.0, {}, availability)
        intervals[active[i - 1]] = LayerInterval(a, b)
        b = a

    if any(interval.width < 1 for interval in intervals.values()):
        return P2Result(False, "zero_width_assignment_forbidden", active, None, 0.0, {}, availability)

    layout = ExecutionLayout(intervals)
    try:
        layout.validate_exact_cover(num_layers)
    except ValueError as exc:
        return P2Result(False, f"invalid_exact_cover:{exc}", active, None, 0.0, {}, availability)

    stage = {u: layout.interval(u).width * system.uav(u).per_layer_latency_s for u in active}
    return P2Result(
        valid=True,
        invalid_reason=None,
        active_uavs=active,
        layout=layout,
        max_stage_latency_s=max(stage.values()) if stage else 0.0,
        stage_latency_by_uav=stage,
        availability=availability,
    )
