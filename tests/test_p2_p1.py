from aerokv2.failure_process import FailureEvent
from aerokv2.p1_new import solve_p1_new
from aerokv2.p1_provisioning import solve_p1_provisioning, worst_case_recovery_latency_s
from aerokv2.p2_reconfiguration import build_availability_index, solve_p2_reconfiguration
from aerokv2.protection_runtime import update_protection_runtime_at_completed_token
from aerokv2.recovery import compute_recovery
from aerokv2.scenario import make_initial_layout, make_initial_ring, make_standard_system
from aerokv2.sim_standard import make_fixed_protection_plan


def test_p2_rejects_unavailable_uniform_fallback():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    # k=1 keeps state availability intentionally narrow.
    plan = make_fixed_protection_plan(system, layout, ring, method="AeroKV", overlap_depth=1, snapshot_period=128)
    runtime = update_protection_runtime_at_completed_token(system, plan, 128).runtime
    alive = set(range(system.num_uavs)) - {5}
    result = solve_p2_reconfiguration(system, plan, runtime, token=128, alive_uavs=alive)
    # A valid exact cover is not guaranteed under narrow state availability.
    # The key invariant is that invalid results do not return a magic fallback layout.
    if not result.valid:
        assert result.layout is None
        assert result.invalid_reason == "no_state_constrained_exact_cover"


def test_p2_valid_after_full_overlap_failure():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    # Full overlap makes every failed shard head recoverable on predecessor.
    plan = make_fixed_protection_plan(system, layout, ring, method="AeroKV", overlap_depth=5, snapshot_period=None)
    runtime = update_protection_runtime_at_completed_token(system, plan, 64).runtime
    fail = FailureEvent(token=64, failed_uav=5)
    alive_before = set(range(system.num_uavs))
    rec = compute_recovery(system, plan, runtime, fail, alive_uavs_before_failure=alive_before)
    assert rec.valid
    alive_after = alive_before - {5}
    result = solve_p2_reconfiguration(
        system,
        plan,
        runtime,
        token=64,
        alive_uavs=alive_after,
        recovered_intervals_by_uav=rec.recovered_intervals_by_uav,
    )
    assert result.valid, result.invalid_reason
    assert result.layout is not None
    result.layout.validate_exact_cover(system.model.num_layers)
    for uav_id, interval in result.layout.intervals.items():
        assert result.availability.can_claim(uav_id, interval)


def test_p1_returns_globally_valid_plan():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    result = solve_p1_provisioning(system, layout, ring, beam_width=128)
    assert result.valid, result.invalid_reason
    assert result.plan is not None
    result.plan.validate_against(system)
    assert result.memory_feasible
    assert result.deadline_met
    for source in result.plan.layout.intervals:
        latency = worst_case_recovery_latency_s(
            system,
            result.plan.layout,
            result.plan.ring,
            source_uav=source,
            overlap_depth=result.plan.head_overlap_depth[source],
            snapshot_period=result.plan.snapshot_period[source],
        )
        assert latency <= system.tau_recover_max_s


def test_p1_new_replans_on_p2_layout():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    plan = make_fixed_protection_plan(system, layout, ring, method="AeroKV", overlap_depth=5, snapshot_period=None)
    runtime = update_protection_runtime_at_completed_token(system, plan, 64).runtime
    fail = FailureEvent(token=64, failed_uav=5)
    alive_before = set(range(system.num_uavs))
    rec = compute_recovery(system, plan, runtime, fail, alive_uavs_before_failure=alive_before)
    p2 = solve_p2_reconfiguration(
        system,
        plan,
        runtime,
        token=64,
        alive_uavs=alive_before - {5},
        recovered_intervals_by_uav=rec.recovered_intervals_by_uav,
    )
    assert p2.valid, p2.invalid_reason
    p1new = solve_p1_new(system, p2, beam_width=128)
    assert p1new.valid, p1new.invalid_reason
    assert p1new.plan is not None
    p1new.plan.validate_against(system)
    assert set(p1new.plan.layout.intervals) == set(p2.active_uavs)
