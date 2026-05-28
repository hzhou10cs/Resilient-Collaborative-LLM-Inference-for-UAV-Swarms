"""P1-new provisioning after P2 reconfiguration.

P1-new rebuilds AeroKV protection on the surviving ring and the P2 execution
layout.  It is a thin, explicit wrapper around the P1 solver so that post-failure
provisioning is not hidden inside the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..specs import ExecutionLayout, LogicalRing, ProtectionPlan, SystemSpec
from .p1_provisioning import P1Result, solve_p1_provisioning
from .p2_reconfiguration import P2Result


@dataclass(frozen=True)
class P1NewResult:
    valid: bool
    invalid_reason: str | None
    plan: ProtectionPlan | None
    p1_result: P1Result | None
    surviving_ring: LogicalRing | None


def solve_p1_new(
    system: SystemSpec,
    p2_result: P2Result,
    *,
    method: str = "AeroKV-P1-new",
    beam_width: int = 256,
    max_candidates_per_source: int | None = None,
) -> P1NewResult:
    """Re-solve AeroKV protection after a valid P2 layout.

    Input must be a valid P2 result.  The new logical ring is the active UAV
    order from P2; this keeps resilience order aligned with the post-failure
    execution order.
    """

    print(
        f"[P1-new][start] p2_valid={p2_result.valid} "
        f"active_uavs={p2_result.active_uavs} method={method}"
    )
    if not p2_result.valid or p2_result.layout is None:
        print(f"[P1-new][failed] reason={p2_result.invalid_reason or 'invalid_p2_result'}")
        return P1NewResult(
            valid=False,
            invalid_reason=p2_result.invalid_reason or "invalid_p2_result",
            plan=None,
            p1_result=None,
            surviving_ring=None,
        )
    if len(p2_result.active_uavs) < 2:
        print("[P1-new][failed] reason=p1_new_requires_at_least_two_surviving_uavs")
        return P1NewResult(
            valid=False,
            invalid_reason="p1_new_requires_at_least_two_surviving_uavs",
            plan=None,
            p1_result=None,
            surviving_ring=None,
        )

    ring = LogicalRing(tuple(p2_result.active_uavs))
    print(f"[P1-new][ring] surviving_ring={ring.uav_ids}")
    for uav_id in p2_result.active_uavs:
        interval = p2_result.layout.interval(uav_id)
        print(f"[P1-new][input layout] uav={uav_id} layers=[{interval.start},{interval.end})")
    p1 = solve_p1_provisioning(
        system,
        p2_result.layout,
        ring,
        method=method,
        beam_width=beam_width,
        max_candidates_per_source=max_candidates_per_source,
    )
    if p1.valid and p1.plan is not None:
        print("[P1-new][success] rebuilt protection on P2 layout")
        for uav_id in p1.plan.layout.intervals:
            interval = p1.plan.layout.interval(uav_id)
            print(
                f"[P1-new][chosen] source_uav={uav_id} shard=[{interval.start},{interval.end}) "
                f"K={p1.plan.head_overlap_depth[uav_id]} T_B={p1.plan.snapshot_period[uav_id]} "
                f"live_owner=pred({uav_id})={p1.plan.ring.pred(uav_id)} "
                f"snapshot_owner=succ({uav_id})={p1.plan.ring.succ(uav_id)}"
            )
    else:
        print(f"[P1-new][failed] reason={p1.invalid_reason}")
    return P1NewResult(
        valid=p1.valid,
        invalid_reason=p1.invalid_reason,
        plan=p1.plan,
        p1_result=p1,
        surviving_ring=ring,
    )


def solve_p1_new_for_layout(
    system: SystemSpec,
    layout: ExecutionLayout,
    active_uavs: tuple[int, ...],
    *,
    method: str = "AeroKV-P1-new",
    beam_width: int = 256,
    max_candidates_per_source: int | None = None,
) -> P1NewResult:
    """Convenience wrapper when the caller already has a post-failure layout."""

    if len(active_uavs) < 2:
        print("[P1-new][failed] reason=p1_new_requires_at_least_two_surviving_uavs")
        return P1NewResult(False, "p1_new_requires_at_least_two_surviving_uavs", None, None, None)
    ring = LogicalRing(active_uavs)
    print(f"[P1-new][start] active_uavs={active_uavs} method={method}")
    print(f"[P1-new][ring] surviving_ring={ring.uav_ids}")
    for uav_id in active_uavs:
        interval = layout.interval(uav_id)
        print(f"[P1-new][input layout] uav={uav_id} layers=[{interval.start},{interval.end})")
    p1 = solve_p1_provisioning(
        system,
        layout,
        ring,
        method=method,
        beam_width=beam_width,
        max_candidates_per_source=max_candidates_per_source,
    )
    if p1.valid and p1.plan is not None:
        print("[P1-new][success] rebuilt protection on provided layout")
        for uav_id in p1.plan.layout.intervals:
            interval = p1.plan.layout.interval(uav_id)
            print(
                f"[P1-new][chosen] source_uav={uav_id} shard=[{interval.start},{interval.end}) "
                f"K={p1.plan.head_overlap_depth[uav_id]} T_B={p1.plan.snapshot_period[uav_id]} "
                f"live_owner=pred({uav_id})={p1.plan.ring.pred(uav_id)} "
                f"snapshot_owner=succ({uav_id})={p1.plan.ring.succ(uav_id)}"
            )
    else:
        print(f"[P1-new][failed] reason={p1.invalid_reason}")
    return P1NewResult(p1.valid, p1.invalid_reason, p1.plan, p1, ring)
