from aerokv.simulation.engine import make_fixed_protection_plan, simulate_fixed_plan_prefailure
from aerokv.simulation.metrics import system_expected_remaining_tokens
from aerokv.specs import ExecutionLayout, LayerInterval, LogicalRing, ModelSpec, SystemSpec, UAVSpec


def make_tiny_case():
    model = ModelSpec(
        num_layers=4,
        hidden_size=4,
        num_heads=1,
        n_est=4,
        bytes_per_value=2,
        model_params_billion=0.000000004,
    )
    uavs = tuple(
        UAVSpec(
            uav_id=i,
            memory_budget_bytes=1_000_000.0,
            initial_energy_j=100.0,
            flight_power_w=10.0,
            inference_power_w=5.0,
            per_layer_latency_s=0.1,
            link_bps=1_000_000.0,
            tx_power_w=2.0,
        )
        for i in range(2)
    )
    system = SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=3.0,
        storage_load_bps=1_000.0,
        snapshot_period_candidates=(2,),
    )
    layout = ExecutionLayout({0: LayerInterval(0, 2), 1: LayerInterval(2, 4)})
    ring = LogicalRing((0, 1))
    plan = make_fixed_protection_plan(system, layout, ring, overlap_depth=1, snapshot_period=2)
    return system, plan


def test_expected_remaining_tokens_are_recorded_per_uav_and_system():
    system, plan = make_tiny_case()
    out = simulate_fixed_plan_prefailure(system, plan, run_id="expected-token-test", max_tokens=2)

    token1 = [r for r in out.uav_trace if r.token == 1]
    assert len(token1) == 2
    assert all(r.expected_remaining_tokens > 0 for r in token1)

    system_value = system_expected_remaining_tokens(token1)
    assert out.token_trace[1].system_expected_remaining_tokens == system_value
    assert out.summary.final_system_expected_remaining_tokens == out.token_trace[-1].system_expected_remaining_tokens
