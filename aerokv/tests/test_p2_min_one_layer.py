"""Regression test: P2 must not assign zero-width intervals to active UAVs."""

from aerokv.optimizers.p2_reconfiguration import solve_p2_reconfiguration
from aerokv.protection_state import update_protection_runtime_at_completed_token
from aerokv.experiments.scenarios import make_initial_layout, make_initial_ring, make_standard_system
from aerokv.baselines import make_aerokv_plan


def test_p2_assigns_at_least_one_layer_to_each_active_uav():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    plan = make_aerokv_plan(system, layout, ring)
    runtime = update_protection_runtime_at_completed_token(system, plan, 64).runtime
    alive = set(range(system.num_uavs)) - {3}
    result = solve_p2_reconfiguration(
        system,
        plan,
        runtime,
        token=64,
        alive_uavs=alive,
        recovered_intervals_by_uav={},
    )
    if result.valid:
        assert result.layout is not None
        assert all(result.layout.interval(u).width >= 1 for u in result.active_uavs)
