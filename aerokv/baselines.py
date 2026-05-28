"""Baseline protection-plan constructors.

Baselines used in the current experiments:

- NP: no pre-protection; failures use hard recovery in ``recovery.py``.
- OO: overlap-only; each holder stores the maximum memory-feasible head of
  its successor's shard, capped by that shard width.
- SO: snapshot-only; each holder stores full-shard KV snapshots of its
  predecessor's entire shard with fixed T_B = 32.
"""

from __future__ import annotations

import math

from .specs import ExecutionLayout, LogicalRing, ProtectionPlan, SystemSpec


def no_protection_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing) -> ProtectionPlan:
    return ProtectionPlan(
        method="NP",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: 0 for u in layout.intervals},
        snapshot_period={u: None for u in layout.intervals},
    )


def _max_overlap_depth_for_source(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing, source: int) -> int:
    """Maximum K for source's head overlap stored on pred(source).

    This is evaluated at the full task horizon, so the overlap allocation is
    memory-feasible for the worst in-task KV footprint.  OO has no snapshots,
    so each holder only needs its native shard plus the successor's overlap.
    """

    owner = ring.pred(source)
    owner_width = layout.interval(owner).width
    source_width = layout.interval(source).width
    per_layer_full_kv = system.model.weight_bytes_per_layer + system.model.n_est * system.model.kv_bytes_per_token_layer
    native_owner_memory = owner_width * per_layer_full_kv
    available = system.uav(owner).memory_budget_bytes - native_owner_memory
    if available <= 0:
        return 0
    return min(source_width, max(0, int(math.floor(available / per_layer_full_kv))))


def overlap_only_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing, k: int | None = None) -> ProtectionPlan:
    """Construct OO.

    If ``k`` is omitted, choose the maximum memory-feasible overlap depth per
    source.  If ``k`` is provided, still cap by the memory-feasible value and the
    source shard width.
    """

    depths: dict[int, int] = {}
    for source in layout.intervals:
        max_k = _max_overlap_depth_for_source(system, layout, ring, source)
        requested = max_k if k is None else min(k, max_k)
        depths[source] = min(layout.interval(source).width, max(0, requested))
    return ProtectionPlan(
        method="OO",
        ring=ring,
        layout=layout,
        head_overlap_depth=depths,
        snapshot_period={u: None for u in layout.intervals},
    )


def snapshot_only_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing, period: int = 32) -> ProtectionPlan:
    """Construct SO with fixed-T_B=32 full-shard KV snapshots by default.

    From holder UAV i's perspective, it snapshots pred(i)'s entire shard.  From
    source UAV i's perspective, the full shard is snapshotted to succ(i).
    """

    return ProtectionPlan(
        method="SO",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: 0 for u in layout.intervals},
        snapshot_period={u: period for u in layout.intervals},
    )
