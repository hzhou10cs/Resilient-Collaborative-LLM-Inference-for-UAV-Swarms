"""Single-failure recovery accounting for AeroKV.

This layer computes recovery from the compact protection runtime. It does not
perform P2 reconfiguration or P1-new rebuilding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .accounting import (
    latest_snapshot_token_for_source,
    live_overlap_layers_for_source,
    snapshot_tail_layers_for_source,
)
from .core import LayerInterval, ProtectionPlan, ProtectionRuntime, SystemSpec
from .failure_process import FailureEvent


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


def compute_recovery(
    system: SystemSpec,
    plan: ProtectionPlan,
    runtime: ProtectionRuntime,
    failure: FailureEvent,
    *,
    alive_uavs_before_failure: set[int],
) -> RecoveryResult:
    """Recover one failed UAV using the current AeroKV protection state.

    A failure at token t is evaluated after token t has completed. The failed
    source's head layers are available on pred(source); its tail layers are
    recovered from the latest snapshot on succ(source) plus replay since that
    snapshot. No P2 or P1-new is applied here.
    """

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
            "snapshot_owner_unavailable",
            live_owner=live_owner,
            snapshot_owner=snapshot_owner,
            live_layers=live_layers,
            tail_layers=tail_layers,
        )

    latest = latest_snapshot_token_for_source(plan, failed, token, runtime)
    if tail_layers > 0 and latest is None:
        return _invalid(
            failure,
            "missing_snapshot_for_tail",
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
        replay_s = replay_tokens * tail_layers * system.uav(snapshot_owner).per_layer_latency_s
    latency_s = load_s + replay_s

    replay_energy: dict[int, float] = {}
    if replay_s > 0.0 and tail_layers > 0:
        replay_energy[snapshot_owner] = system.uav(snapshot_owner).inference_power_w * replay_s

    shard = plan.layout.interval(failed)
    live_interval = LayerInterval(shard.start, shard.start + live_layers)
    tail_interval = LayerInterval(shard.start + live_layers, shard.end)

    recovered: dict[int, int] = {}
    recovered_intervals: dict[int, list[LayerInterval]] = {}
    if live_layers > 0:
        recovered[live_owner] = recovered.get(live_owner, 0) + live_layers
        recovered_intervals.setdefault(live_owner, []).append(live_interval)
    if tail_layers > 0:
        recovered[snapshot_owner] = recovered.get(snapshot_owner, 0) + tail_layers
        recovered_intervals.setdefault(snapshot_owner, []).append(tail_interval)

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
