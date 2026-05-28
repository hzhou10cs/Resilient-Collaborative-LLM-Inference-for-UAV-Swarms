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
layout with the minimum total chain latency.  Reconfiguration data movement is
not modeled here; illegal movement is rejected rather than hidden behind a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..accounting import (
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    runtime_per_layer_latency_s,
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
    total_chain_latency_s: float
    max_stage_latency_s: float
    stage_latencies_s: tuple[float, ...]
    compute_latencies_s: tuple[float, ...]
    activation_forward_latencies_s: tuple[float, ...]
    per_layer_latency_by_uav_s: Mapping[int, float]
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
        native = plan.layout.interval(uav_id)
        out[uav_id].update(_layers(native))
        print(
            f"[P2][availability source] uav={uav_id} source=native "
            f"layers=[{native.start},{native.end}) reason=uav_survived_with_own_shard"
        )

        # Live-overlap head held for successor.
        succ = plan.ring.succ(uav_id)
        if succ in plan.layout.intervals:
            head = _head_interval(plan, succ)
            out[uav_id].update(_layers(head))
            print(
                f"[P2][availability source] uav={uav_id} source=live_overlap "
                f"protected_source={succ} layers=[{head.start},{head.end}) "
                f"reason=uav_holds_successor_head"
            )

        # Snapshot tail held for predecessor.  It is claimable only if there is
        # an actual snapshot for that source in the runtime state.
        pred = plan.ring.pred(uav_id)
        if pred in plan.layout.intervals and plan.snapshot_period[pred] is not None:
            tail = _tail_interval(plan, pred)
            latest = latest_snapshot_token_for_source(plan, pred, token, runtime)
            if latest is not None and latest <= token and tail.width > 0:
                out[uav_id].update(_layers(tail))
                print(
                    f"[P2][availability source] uav={uav_id} source=snapshot_tail "
                    f"protected_source={pred} layers=[{tail.start},{tail.end}) "
                    f"latest_snapshot_token={latest} reason=uav_holds_predecessor_snapshot_tail"
                )
            else:
                print(
                    f"[P2][availability rejected] uav={uav_id} source=snapshot_tail "
                    f"protected_source={pred} layers=[{tail.start},{tail.end}) "
                    f"latest_snapshot_token={latest} reason=no_claimable_snapshot_tail"
                )
        else:
            print(
                f"[P2][availability rejected] uav={uav_id} source=snapshot_tail "
                f"protected_source={pred} reason=no_snapshot_period_or_missing_predecessor"
            )

        # Intervals already materialized by the recovery step.
        for interval in recovered_intervals_by_uav.get(uav_id, ()):
            out[uav_id].update(_layers(interval))
            print(
                f"[P2][availability source] uav={uav_id} source=recovery "
                f"layers=[{interval.start},{interval.end}) reason=recovery_materialized_interval"
            )

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


def activation_forward_latency_s(system: SystemSpec, source_uav: int, next_uav: int | None) -> float:
    """TX time for forwarding one token's activation to the next chain stage."""

    if next_uav is None:
        return 0.0
    _ = next_uav
    return system.model.activation_bytes * 8.0 / system.uav(source_uav).link_bps


def p2_stage_latency_s(
    system: SystemSpec,
    uav_id: int,
    interval: LayerInterval,
    next_uav: int | None,
    *,
    energy_j: float | None = None,
) -> tuple[float, float, float, float]:
    """Return stage, compute, activation-forward, and per-layer latency."""

    per_layer = runtime_per_layer_latency_s(system, uav_id, energy_j)
    compute = interval.width * per_layer
    forward = activation_forward_latency_s(system, uav_id, next_uav)
    return compute + forward, compute, forward, per_layer


def solve_p2_reconfiguration(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    *,
    token: int,
    alive_uavs: set[int],
    recovered_intervals_by_uav: Mapping[int, tuple[LayerInterval, ...]] | None = None,
    energy_by_uav_j: Mapping[int, float] | None = None,
) -> P2Result:
    """Solve a state-constrained contiguous P2 layout.

    Objective: minimize total per-token chain latency among surviving UAVs.
    Failure: returns valid=False; it does not emit an unconstrained fallback.
    """

    print(
        f"[P2][start] token={token} alive_uavs={tuple(sorted(alive_uavs))} "
        f"failed_uavs={tuple(u for u in plan.ring.uav_ids if u not in alive_uavs)}"
    )
    if not alive_uavs:
        availability = AvailabilityIndex({})
        print("[P2][failed] reason=no_alive_uavs")
        return P2Result(False, "no_alive_uavs", tuple(), None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability)

    availability = build_availability_index(
        system,
        plan,
        runtime,
        token=token,
        alive_uavs=alive_uavs,
        recovered_intervals_by_uav=recovered_intervals_by_uav,
    )
    print("[P2][availability] A surviving UAV can claim only layers listed here.")
    for uav_id in sorted(availability.available_layers_by_uav):
        layers = sorted(availability.available_layers_by_uav[uav_id])
        print(f"[P2][availability] uav={uav_id} layers={layers}")
    if recovered_intervals_by_uav:
        for uav_id, intervals in sorted(recovered_intervals_by_uav.items()):
            text = ", ".join(f"[{iv.start},{iv.end})" for iv in intervals)
            print(f"[P2][recovered intervals] uav={uav_id} intervals={text}")
    active = _active_uavs_in_layer_order(plan, alive_uavs)
    print(f"[P2][active order] active_uavs={active}")
    if not active:
        print("[P2][failed] reason=no_active_uavs_in_layout")
        return P2Result(
            False, "no_active_uavs_in_layout", tuple(), None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability
        )

    num_layers = system.model.num_layers
    if len(active) > num_layers:
        print(
            "[P2][failed] reason=not_enough_layers_for_min_one_layer_per_active_uav "
            f"active_count={len(active)} num_layers={num_layers}"
        )
        return P2Result(
            False,
            "not_enough_layers_for_min_one_layer_per_active_uav",
            active,
            None,
            0.0,
            0.0,
            tuple(),
            tuple(),
            tuple(),
            {},
            {},
            availability,
        )
    inf = float("inf")

    # dp[i][b] is best total-chain-latency after assigning layers [0,b) to the
    # first i active UAVs.  Parent stores the previous boundary a.  Because
    # every active UAV must receive at least one layer, feasible states satisfy
    # b >= i and each transition interval [a,b) has width >= 1.
    dp = [[inf] * (num_layers + 1) for _ in range(len(active) + 1)]
    parent: list[list[int | None]] = [[None] * (num_layers + 1) for _ in range(len(active) + 1)]
    dp[0][0] = 0.0

    for i, uav_id in enumerate(active, start=1):
        next_uav = active[i] if i < len(active) else None
        energy_j = None if energy_by_uav_j is None else energy_by_uav_j.get(uav_id)
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
                    print(
                        f"[P2][dp rejected] step={i}/{len(active)} uav={uav_id} "
                        f"claim=[{a},{b}) reason=state_unavailable "
                        f"available_layers={sorted(availability.available_layers_by_uav.get(uav_id, frozenset()))}"
                    )
                    continue
                if not _memory_ok_for_assignment(system, uav_id, interval, token):
                    print(
                        f"[P2][dp rejected] step={i} uav={uav_id} "
                        f"interval=[{a},{b}) reason=memory_budget"
                    )
                    continue
                stage, compute, forward, per_layer = p2_stage_latency_s(
                    system, uav_id, interval, next_uav, energy_j=energy_j
                )
                value = dp[i - 1][a] + stage
                if value < dp[i][b]:
                    previous = dp[i][b]
                    dp[i][b] = value
                    parent[i][b] = a
                    previous_text = "inf" if previous == inf else f"{previous:.6f}"
                    print(
                        f"[P2][dp update] step={i}/{len(active)} uav={uav_id} "
                        f"claim=[{a},{b}) prev_boundary={a} end_boundary={b} "
                        f"compute_latency_s={compute:.6f} activation_forward_latency_s={forward:.6f} "
                        f"per_layer_latency_s={per_layer:.9f} stage_latency_s={stage:.6f} "
                        f"total_chain_latency_s={value:.6f} "
                        f"previous_best={previous_text}"
                    )
                else:
                    print(
                        f"[P2][dp not chosen] step={i}/{len(active)} uav={uav_id} "
                        f"claim=[{a},{b}) reason=not_better "
                        f"candidate_total_chain_latency_s={value:.6f} "
                        f"current_best_for_boundary_s={dp[i][b]:.6f}"
                    )

    if dp[len(active)][num_layers] == inf:
        print("[P2][failed] reason=no_state_constrained_exact_cover")
        return P2Result(
            False, "no_state_constrained_exact_cover", active, None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability
        )

    intervals: dict[int, LayerInterval] = {}
    b = num_layers
    for i in range(len(active), 0, -1):
        a = parent[i][b]
        if a is None:
            print("[P2][failed] reason=missing_dp_parent")
            return P2Result(
                False, "missing_dp_parent", active, None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability
            )
        intervals[active[i - 1]] = LayerInterval(a, b)
        print(
            f"[P2][backtrack] step={i} uav={active[i - 1]} "
            f"assigned=[{a},{b}) parent_boundary={a}"
        )
        b = a

    if any(interval.width < 1 for interval in intervals.values()):
        print("[P2][failed] reason=zero_width_assignment_forbidden")
        return P2Result(
            False, "zero_width_assignment_forbidden", active, None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability
        )

    layout = ExecutionLayout(intervals)
    try:
        layout.validate_exact_cover(num_layers)
    except ValueError as exc:
        print(f"[P2][failed] reason=invalid_exact_cover detail={exc}")
        return P2Result(
            False, f"invalid_exact_cover:{exc}", active, None, 0.0, 0.0, tuple(), tuple(), tuple(), {}, {}, availability
        )

    stage: dict[int, float] = {}
    compute_by_uav: dict[int, float] = {}
    forward_by_uav: dict[int, float] = {}
    per_layer_by_uav: dict[int, float] = {}
    for idx, u in enumerate(active):
        next_uav = active[idx + 1] if idx + 1 < len(active) else None
        energy_j = None if energy_by_uav_j is None else energy_by_uav_j.get(u)
        stage[u], compute_by_uav[u], forward_by_uav[u], per_layer_by_uav[u] = p2_stage_latency_s(
            system,
            u,
            layout.interval(u),
            next_uav,
            energy_j=energy_j,
        )
    stage_latencies = tuple(stage[u] for u in active)
    compute_latencies = tuple(compute_by_uav[u] for u in active)
    forward_latencies = tuple(forward_by_uav[u] for u in active)
    total_chain_latency = sum(stage_latencies)
    max_stage_latency = max(stage_latencies) if stage_latencies else 0.0
    print("[P2][success] state-constrained exact cover found")
    for uav_id in active:
        interval = layout.interval(uav_id)
        can_claim = availability.can_claim(uav_id, interval)
        print(
            f"[P2][stage] survivor={uav_id} layers=[{interval.start},{interval.end}) "
            f"width={interval.width} per_layer_latency_s={per_layer_by_uav[uav_id]:.9f} "
            f"compute_latency_s={compute_by_uav[uav_id]:.6f} "
            f"activation_forward_latency_s={forward_by_uav[uav_id]:.6f} "
            f"stage_latency_s={stage[uav_id]:.6f} can_claim={can_claim}"
        )
        print(
            f"[P2][claim evidence] uav={uav_id} succeeds because every layer in "
            f"[{interval.start},{interval.end}) is in availability[{uav_id}]="
            f"{sorted(availability.available_layers_by_uav[uav_id])}"
        )
    print(f"[P2][objective] total_chain_latency_s={total_chain_latency:.6f}")
    print(f"[P2][diagnostic] max_stage_latency_s={max_stage_latency:.6f}")
    print(f"[P2][diagnostic] stage_latencies_s={stage_latencies}")
    print(f"[P2][diagnostic] compute_latencies_s={compute_latencies}")
    print(f"[P2][diagnostic] activation_forward_latencies_s={forward_latencies}")
    return P2Result(
        valid=True,
        invalid_reason=None,
        active_uavs=active,
        layout=layout,
        total_chain_latency_s=total_chain_latency,
        max_stage_latency_s=max_stage_latency,
        stage_latencies_s=stage_latencies,
        compute_latencies_s=compute_latencies,
        activation_forward_latencies_s=forward_latencies,
        per_layer_latency_by_uav_s=per_layer_by_uav,
        stage_latency_by_uav=stage,
        availability=availability,
    )
