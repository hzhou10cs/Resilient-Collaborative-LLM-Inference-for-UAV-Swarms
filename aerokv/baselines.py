"""Baseline protection-plan constructors."""

from __future__ import annotations

from .specs import ExecutionLayout, LogicalRing, ProtectionPlan, SystemSpec


def no_protection_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing) -> ProtectionPlan:
    return ProtectionPlan(
        method="NP",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: 0 for u in layout.intervals},
        snapshot_period={u: None for u in layout.intervals},
    )


def overlap_only_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing, k: int) -> ProtectionPlan:
    return ProtectionPlan(
        method="OO",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: min(k, layout.interval(u).width) for u in layout.intervals},
        snapshot_period={u: None for u in layout.intervals},
    )


def snapshot_only_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing, period: int) -> ProtectionPlan:
    return ProtectionPlan(
        method="SO",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: 0 for u in layout.intervals},
        snapshot_period={u: period for u in layout.intervals},
    )


def full_mirror_plan(system: SystemSpec, layout: ExecutionLayout, ring: LogicalRing) -> ProtectionPlan:
    return ProtectionPlan(
        method="Ideal Full Mirror",
        ring=ring,
        layout=layout,
        head_overlap_depth={u: layout.interval(u).width for u in layout.intervals},
        snapshot_period={u: 1 for u in layout.intervals},
        full_mirror=True,
    )
