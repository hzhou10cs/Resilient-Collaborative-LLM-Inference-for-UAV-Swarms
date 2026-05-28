"""P1 provisioning for AeroKV.

The solver is intentionally conservative: it never returns a plan that violates
memory or worst-case recovery deadline constraints.  It uses deadline-guided
candidate generation and a bounded beam search over complete joint plans.  This
keeps the implementation inspectable for the standard experiment while avoiding
the earlier per-UAV independent greedy planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..accounting import (
    average_snapshot_tx_energy_per_token_j,
    live_overlap_layers_for_source,
    memory_by_uav_bytes,
    memory_feasible,
    protected_pipeline_latency_s,
    steady_state_energy_per_token_by_uav_j,
    runtime_inference_power_w,
    runtime_per_layer_latency_s,
)
from ..specs import ExecutionLayout, LogicalRing, ProtectionPlan, SystemSpec
from ..protection_state import update_protection_runtime_at_completed_token


@dataclass(frozen=True)
class P1Candidate:
    source_uav: int
    overlap_depth: int
    snapshot_period: int | None
    worst_recovery_latency_s: float
    local_score_j_per_token: float


@dataclass(frozen=True)
class P1Result:
    valid: bool
    invalid_reason: str | None
    plan: ProtectionPlan | None
    objective_min_lifetime_s: float
    max_energy_per_token_j: float
    max_memory_bytes: float
    deadline_met: bool
    memory_feasible: bool
    candidates_by_source: Mapping[int, tuple[P1Candidate, ...]]


def _candidate_periods(system: SystemSpec) -> tuple[int, ...]:
    return tuple(p for p in system.snapshot_period_candidates if p <= system.model.n_est)


def worst_case_recovery_latency_s(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    source_uav: int,
    overlap_depth: int,
    snapshot_period: int | None,
) -> float:
    """Worst-case latency to recover source_uav's shard under k/T_B."""

    shard = layout.interval(source_uav)
    k = min(overlap_depth, shard.width)
    tail_layers = max(0, shard.width - k)
    if tail_layers == 0:
        return 0.0
    if snapshot_period is None:
        return float("inf")
    succ = ring.succ(source_uav)
    load_s = tail_layers * system.model.weight_bytes_per_layer * 8.0 / system.storage_load_bps
    replay_s = (snapshot_period - 1) * tail_layers * runtime_per_layer_latency_s(system, succ)
    return load_s + replay_s


def _candidate_local_score_j_per_token(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    source_uav: int,
    overlap_depth: int,
    snapshot_period: int | None,
) -> float:
    """Local protection cost used only for candidate ordering within beam search."""

    pred = ring.pred(source_uav)
    succ = ring.succ(source_uav)
    shard_width = layout.interval(source_uav).width
    k = min(overlap_depth, shard_width)
    tail = max(0, shard_width - k)
    overlap_compute = runtime_inference_power_w(system, pred) * k * runtime_per_layer_latency_s(system, pred)
    tx = 0.0
    if snapshot_period is not None and tail > 0:
        tx = system.uav(source_uav).tx_power_w * tail * system.model.kv_bytes_per_token_layer * 8.0 / system.uav(source_uav).link_bps
    # Tiny term prefers longer periods only among equal-energy candidates.
    period_term = 0.0 if snapshot_period is None else 1e-9 / snapshot_period
    # succ is referenced so the source/succ direction remains explicit in this function.
    _ = succ
    return overlap_compute + tx + period_term


def generate_p1_candidates(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    source_uav: int,
) -> tuple[P1Candidate, ...]:
    """Generate deadline-feasible candidates for one source shard."""

    periods = _candidate_periods(system)
    shard_width = layout.interval(source_uav).width
    out: list[P1Candidate] = []
    for k in range(shard_width + 1):
        if k == shard_width:
            latency = 0.0
            out.append(
                P1Candidate(
                    source_uav=source_uav,
                    overlap_depth=k,
                    snapshot_period=None,
                    worst_recovery_latency_s=latency,
                    local_score_j_per_token=_candidate_local_score_j_per_token(
                        system, layout, ring, source_uav=source_uav, overlap_depth=k, snapshot_period=None
                    ),
                )
            )
            continue
        for period in periods:
            latency = worst_case_recovery_latency_s(
                system,
                layout,
                ring,
                source_uav=source_uav,
                overlap_depth=k,
                snapshot_period=period,
            )
            if latency > system.tau_recover_max_s:
                continue
            out.append(
                P1Candidate(
                    source_uav=source_uav,
                    overlap_depth=k,
                    snapshot_period=period,
                    worst_recovery_latency_s=latency,
                    local_score_j_per_token=_candidate_local_score_j_per_token(
                        system, layout, ring, source_uav=source_uav, overlap_depth=k, snapshot_period=period
                    ),
                )
            )
    out.sort(key=lambda c: (c.local_score_j_per_token, c.worst_recovery_latency_s, c.overlap_depth))
    return tuple(out)


def _build_plan(
    method: str,
    layout: ExecutionLayout,
    ring: LogicalRing,
    choices: Mapping[int, P1Candidate],
) -> ProtectionPlan:
    return ProtectionPlan(
        method=method,
        layout=layout,
        ring=ring,
        head_overlap_depth={u: choices[u].overlap_depth for u in layout.intervals},
        snapshot_period={u: choices[u].snapshot_period for u in layout.intervals},
    )


def _evaluate_complete_plan(system: SystemSpec, plan: ProtectionPlan) -> tuple[bool, bool, float, float, float]:
    """Return memory_ok, deadline_ok, min_lifetime_s, max_ept, max_memory."""

    runtime = update_protection_runtime_at_completed_token(system, plan, system.model.n_est).runtime
    mem = memory_by_uav_bytes(system, plan, system.model.n_est, runtime)
    memory_ok = all(mem[u] <= system.uav(u).memory_budget_bytes for u in mem)
    deadline_ok = True
    for u in plan.layout.intervals:
        latency = worst_case_recovery_latency_s(
            system,
            plan.layout,
            plan.ring,
            source_uav=u,
            overlap_depth=live_overlap_layers_for_source(plan, u),
            snapshot_period=plan.snapshot_period[u],
        )
        if latency > system.tau_recover_max_s:
            deadline_ok = False
            break

    latency = protected_pipeline_latency_s(system, plan).pipeline_latency_s
    ept = steady_state_energy_per_token_by_uav_j(system, plan)
    max_energy = max((v.total_j for v in ept.values()), default=0.0)
    lifetimes = []
    for u, breakdown in ept.items():
        if breakdown.total_j <= 0:
            lifetimes.append(float("inf"))
        else:
            lifetimes.append(system.uav(u).initial_energy_j / breakdown.total_j * latency)
    min_life = min(lifetimes) if lifetimes else 0.0
    return memory_ok, deadline_ok, min_life, max_energy, max(mem.values()) if mem else 0.0


def solve_p1_provisioning(
    system: SystemSpec,
    layout: ExecutionLayout,
    ring: LogicalRing,
    *,
    method: str = "AeroKV",
    beam_width: int = 256,
    max_candidates_per_source: int | None = None,
) -> P1Result:
    """Solve P1 with bounded joint search.

    The search state is a partial assignment of candidates to source UAVs.  Beam
    ranking uses the maximum accumulated local protection cost to avoid selecting
    candidates independently.  Every returned plan is globally validated for
    memory and deadline constraints at n_est.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    layout.validate_exact_cover(system.model.num_layers)
    sources = tuple(layout.intervals.keys())
    candidates: dict[int, tuple[P1Candidate, ...]] = {}
    for source in sources:
        cs = generate_p1_candidates(system, layout, ring, source_uav=source)
        if max_candidates_per_source is not None:
            cs = cs[:max_candidates_per_source]
        candidates[source] = cs
        if not cs:
            return P1Result(
                valid=False,
                invalid_reason=f"no_deadline_feasible_candidate_for_uav_{source}",
                plan=None,
                objective_min_lifetime_s=0.0,
                max_energy_per_token_j=0.0,
                max_memory_bytes=0.0,
                deadline_met=False,
                memory_feasible=False,
                candidates_by_source=candidates,
            )

    # Beam item: (rough_score, choices)
    beam: list[tuple[float, dict[int, P1Candidate]]] = [(0.0, {})]
    ordered_sources = tuple(sorted(sources, key=lambda u: layout.interval(u).start))
    for source in ordered_sources:
        expanded: list[tuple[float, dict[int, P1Candidate]]] = []
        for score, choices in beam:
            for cand in candidates[source]:
                nxt = dict(choices)
                nxt[source] = cand
                # Penalize maximum local burden and sum burden.  This keeps the
                # search joint without requiring MILP machinery.
                local_values = [c.local_score_j_per_token for c in nxt.values()]
                rough = max(local_values) + 0.01 * sum(local_values)
                expanded.append((rough, nxt))
        expanded.sort(key=lambda item: item[0])
        beam = expanded[:beam_width]

    best_plan: ProtectionPlan | None = None
    best_min_life = -1.0
    best_max_energy = 0.0
    best_max_memory = 0.0
    best_memory_ok = False
    best_deadline_ok = False

    for _, choices in beam:
        if set(choices) != set(sources):
            continue
        plan = _build_plan(method, layout, ring, choices)
        plan.validate_against(system)
        memory_ok, deadline_ok, min_life, max_energy, max_memory = _evaluate_complete_plan(system, plan)
        if not memory_ok or not deadline_ok:
            continue
        if min_life > best_min_life:
            best_plan = plan
            best_min_life = min_life
            best_max_energy = max_energy
            best_max_memory = max_memory
            best_memory_ok = memory_ok
            best_deadline_ok = deadline_ok

    if best_plan is None:
        return P1Result(
            valid=False,
            invalid_reason="no_global_memory_and_deadline_feasible_plan_in_beam",
            plan=None,
            objective_min_lifetime_s=0.0,
            max_energy_per_token_j=0.0,
            max_memory_bytes=0.0,
            deadline_met=False,
            memory_feasible=False,
            candidates_by_source=candidates,
        )

    return P1Result(
        valid=True,
        invalid_reason=None,
        plan=best_plan,
        objective_min_lifetime_s=best_min_life,
        max_energy_per_token_j=best_max_energy,
        max_memory_bytes=best_max_memory,
        deadline_met=best_deadline_ok,
        memory_feasible=best_memory_ok,
        candidates_by_source=candidates,
    )
