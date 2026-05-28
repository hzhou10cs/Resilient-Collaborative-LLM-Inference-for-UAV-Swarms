from aerokv.experiments.scenarios import make_initial_layout, make_initial_ring, make_standard_system
from aerokv.optimizers.p1_provisioning import solve_p1_provisioning


def test_p1_returns_globally_valid_plan_or_explicit_invalid():
    system = make_standard_system(seed=2026)
    layout = make_initial_layout(system)
    ring = make_initial_ring(system)
    result = solve_p1_provisioning(system, layout, ring, beam_width=32)
    if result.valid:
        assert result.plan is not None
        assert result.memory_feasible
        assert result.deadline_met
        assert result.objective_min_lifetime_s > 0
    else:
        assert result.invalid_reason is not None
