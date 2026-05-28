"""Failure recovery accounting for the four experiment methods.

Recovery semantics:

- NP: no pre-protection.  The failed shard is hard-recovered by splitting the
  disconnected layers between the predecessor and successor, balanced by their
  per-layer latency.
- OO: the predecessor already has the failed shard's covered head overlap.  Any
  uncovered tail is hard-recovered on the successor.
- SO: the successor has a token-fresh full-KV snapshot when T_B = 1; it loads
  the failed shard weights and replays only if the snapshot is stale.
- AeroKV: the predecessor owns the live head; the successor recovers the tail
  from Boundary Snapshot and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .accounting import (
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    snapshot_tail_layers_for_source,
    runtime_inference_power_w,
    runtime_per_layer_latency_s,
)
from .specs import LayerInterval, ProtectionPlan, ProtectionRuntime, SystemSpec
from .simulation.events import FailureEvent


@dataclass(frozen=True)
class RecoveryResult:
    failure: FailureEvent
    valid: bool
    invalid_reason: str | None
    live_owner_uav: int | None
    snapshot_owner_uav: int | None
    live_head_layers: int
    snapshot_tail_layers: int
    latest_snapshot_token: int | None
    replay_tokens: int
    recovery_latency_s: float
    deadline_met: bool
    replay_compute_energy_by_uav_j: Mapping[int, float]
    recovered_exec_layers_by_uav: Mapping[int, int]
    recovered_intervals_by_uav: Mapping[int, tuple[LayerInterval, ...]]


def _invalid(
    failure: FailureEvent,
    reason: str,
    *,
    live_owner: int | None,
    snapshot_owner: int | None,
    live_layers: int,
    tail_layers: int,
    latest: int | None = None,
) -> RecoveryResult:
    return RecoveryResult(
        failure=failure,
        valid=False,
        invalid_reason=reason,
        live_owner_uav=live_owner,
        snapshot_owner_uav=snapshot_owner,
        live_head_layers=live_layers,
        snapshot_tail_layers=tail_layers,
        latest_snapshot_token=latest,
        replay_tokens=0,
        recovery_latency_s=0.0,
        deadline_met=False,
        replay_compute_energy_by_uav_j={},
        recovered_exec_layers_by_uav={},
        recovered_intervals_by_uav={},
    )


def _add_recovered_interval(
    recovered_layers: dict[int, int],
    recovered_intervals: dict[int, list[LayerInterval]],
    owner: int,
    interval: LayerInterval,
) -> None:
    if interval.width <= 0:
        return
    recovered_layers[owner] = recovered_layers.get(owner, 0) + interval.width
    recovered_intervals.setdefault(owner, []).append(interval)


def _balanced_split(width: int, left_latency_s: float, right_latency_s: float) -> int:
    """Return layers assigned to the left side for balanced hard recovery."""

    if width <= 0:
        return 0
    if width == 1:
        return 1 if left_latency_s <= right_latency_s else 0
    best_left = 1
    best_cost = float("inf")
    for left_layers in range(1, width):
        right_layers = width - left_layers
        cost = max(left_layers * left_latency_s, right_layers * right_latency_s)
        if cost < best_cost:
            best_cost = cost
            best_left = left_layers
    return best_left


def _hard_recovery_np(
    system: SystemSpec,
    plan: ProtectionPlan,
    failure: FailureEvent,
    *,
    alive_uavs_before_failure: set[int],
) -> RecoveryResult:
    """NP hard recovery by splitting the failed shard across two neighbors."""

    failed = failure.failed_uav
    token = failure.token
    pred = plan.ring.pred(failed)
    succ = plan.ring.succ(failed)
    shard = plan.layout.interval(failed)

    if pred not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "hard_recovery_predecessor_unavailable",
            live_owner=pred,
            snapshot_owner=succ,
            live_layers=0,
            tail_layers=shard.width,
        )
    if succ not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "hard_recovery_successor_unavailable",
            live_owner=pred,
            snapshot_owner=succ,
            live_layers=0,
            tail_layers=shard.width,
        )

    left_layers = _balanced_split(shard.width, runtime_per_layer_latency_s(system, pred), runtime_per_layer_latency_s(system, succ))
    right_layers = shard.width - left_layers
    split = shard.start + left_layers
    left_interval = LayerInterval(shard.start, split)
    right_interval = LayerInterval(split, shard.end)

    load_left_s = left_layers * system.model.weight_bytes_per_layer * 8.0 / system.storage_load_bps
    load_right_s = right_layers * system.model.weight_bytes_per_layer * 8.0 / system.storage_load_bps
    replay_left_s = token * left_layers * runtime_per_layer_latency_s(system, pred)
    replay_right_s = token * right_layers * runtime_per_layer_latency_s(system, succ)
    latency_s = max(load_left_s, load_right_s) + max(replay_left_s, replay_right_s)

    replay_energy: dict[int, float] = {}
    if replay_left_s > 0:
        replay_energy[pred] = runtime_inference_power_w(system, pred) * replay_left_s
    if replay_right_s > 0:
        replay_energy[succ] = runtime_inference_power_w(system, succ) * replay_right_s

    recovered: dict[int, int] = {}
    recovered_intervals: dict[int, list[LayerInterval]] = {}
    _add_recovered_interval(recovered, recovered_intervals, pred, left_interval)
    _add_recovered_interval(recovered, recovered_intervals, succ, right_interval)

    return RecoveryResult(
        failure=failure,
        valid=True,
        invalid_reason=None,
        live_owner_uav=pred if left_layers > 0 else None,
        snapshot_owner_uav=succ if right_layers > 0 else None,
        live_head_layers=0,
        snapshot_tail_layers=0,
        latest_snapshot_token=None,
        replay_tokens=token,
        recovery_latency_s=latency_s,
        deadline_met=latency_s <= system.tau_recover_max_s,
        replay_compute_energy_by_uav_j=replay_energy,
        recovered_exec_layers_by_uav=recovered,
        recovered_intervals_by_uav={u: tuple(v) for u, v in recovered_intervals.items()},
    )


def _hard_recovery_oo_tail(
    system: SystemSpec,
    plan: ProtectionPlan,
    failure: FailureEvent,
    *,
    alive_uavs_before_failure: set[int],
    live_owner: int,
    snapshot_owner: int,
    live_layers: int,
    tail_layers: int,
) -> RecoveryResult:
    """OO recovery: covered head is live; uncovered tail is replayed on succ."""

    failed = failure.failed_uav
    token = failure.token
    shard = plan.layout.interval(failed)
    if live_layers > 0 and live_owner not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "live_overlap_owner_unavailable",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
        )
    if tail_layers > 0 and snapshot_owner not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "hard_recovery_successor_unavailable",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
        )

    live_interval = LayerInterval(shard.start, shard.start + live_layers)
    tail_interval = LayerInterval(shard.start + live_layers, shard.end)
    load_s = tail_layers * system.model.weight_bytes_per_layer * 8.0 / system.storage_load_bps
    replay_s = token * tail_layers * runtime_per_layer_latency_s(system, snapshot_owner)
    latency_s = load_s + replay_s

    replay_energy: dict[int, float] = {}
    if replay_s > 0:
        replay_energy[snapshot_owner] = runtime_inference_power_w(system, snapshot_owner) * replay_s

    recovered: dict[int, int] = {}
    recovered_intervals: dict[int, list[LayerInterval]] = {}
    _add_recovered_interval(recovered, recovered_intervals, live_owner, live_interval)
    _add_recovered_interval(recovered, recovered_intervals, snapshot_owner, tail_interval)

    return RecoveryResult(
        failure=failure,
        valid=True,
        invalid_reason=None,
        live_owner_uav=live_owner if live_layers > 0 else None,
        snapshot_owner_uav=snapshot_owner if tail_layers > 0 else None,
        live_head_layers=live_layers,
        snapshot_tail_layers=0,
        latest_snapshot_token=None,
        replay_tokens=token if tail_layers > 0 else 0,
        recovery_latency_s=latency_s,
        deadline_met=latency_s <= system.tau_recover_max_s,
        replay_compute_energy_by_uav_j=replay_energy,
        recovered_exec_layers_by_uav=recovered,
        recovered_intervals_by_uav={u: tuple(v) for u, v in recovered_intervals.items()},
    )


def compute_recovery(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    failure: FailureEvent,
    *,
    alive_uavs_before_failure: set[int],
) -> RecoveryResult:
    """Recover one failed UAV using the method-specific recovery semantics."""

    failure.validate_against(system)
    failed = failure.failed_uav
    token = failure.token

    live_owner = plan.ring.pred(failed)
    snapshot_owner = plan.ring.succ(failed)
    live_layers = live_overlap_layers_for_source(plan, failed)
    tail_layers = snapshot_tail_layers_for_source(plan, failed)

    if failed not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "uav_already_failed",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
        )

    if plan.method == "NP":
        return _hard_recovery_np(system, plan, failure, alive_uavs_before_failure=alive_uavs_before_failure)

    if live_layers > 0 and live_owner not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "live_overlap_owner_unavailable",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
        )

    latest = latest_snapshot_token_for_source(plan, failed, token, runtime)
    if tail_layers > 0 and latest is None:
        if plan.method == "OO":
            return _hard_recovery_oo_tail(
                system,
                plan,
                failure,
                alive_uavs_before_failure=alive_uavs_before_failure,
                live_owner=live_owner,
                snapshot_owner=snapshot_owner,
                live_layers=live_layers,
                tail_layers=tail_layers,
            )
        return _invalid(
            failure,
            "missing_snapshot_for_tail",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
            latest=latest,
        )

    if tail_layers > 0 and snapshot_owner not in alive_uavs_before_failure:
        return _invalid(
            failure,
            "snapshot_owner_unavailable",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
            latest=latest,
        )

    replay_tokens = 0 if latest is None else max(0, token - latest)
    if tail_layers == 0:
        load_s = 0.0
        replay_s = 0.0
    else:
        load_s = tail_layers * system.model.weight_bytes_per_layer * 8.0 / system.storage_load_bps
        replay_s = replay_tokens * tail_layers * runtime_per_layer_latency_s(system, snapshot_owner)
    latency_s = load_s + replay_s

    replay_energy: dict[int, float] = {}
    if replay_s > 0.0 and tail_layers > 0:
        replay_energy[snapshot_owner] = runtime_inference_power_w(system, snapshot_owner) * replay_s

    shard = plan.layout.interval(failed)
    live_interval = LayerInterval(shard.start, shard.start + live_layers)
    tail_interval = LayerInterval(shard.start + live_layers, shard.end)

    recovered: dict[int, int] = {}
    recovered_intervals: dict[int, list[LayerInterval]] = {}
    _add_recovered_interval(recovered, recovered_intervals, live_owner, live_interval)
    _add_recovered_interval(recovered, recovered_intervals, snapshot_owner, tail_interval)

    return RecoveryResult(
        failure=failure,
        valid=True,
        invalid_reason=None,
        live_owner_uav=live_owner if live_layers > 0 else None,
        snapshot_owner_uav=snapshot_owner if tail_layers > 0 else None,
        live_head_layers=live_layers,
        snapshot_tail_layers=tail_layers,
        latest_snapshot_token=latest,
        replay_tokens=replay_tokens,
        recovery_latency_s=latency_s,
        deadline_met=latency_s <= system.tau_recover_max_s,
        replay_compute_energy_by_uav_j=replay_energy,
        recovered_exec_layers_by_uav=recovered,
        recovered_intervals_by_uav={u: tuple(v) for u, v in recovered_intervals.items()},
    )
