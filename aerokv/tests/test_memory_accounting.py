import math

from aerokv.accounting import (
    average_snapshot_tx_bytes_per_token,
    average_snapshot_tx_energy_per_token_j,
    memory_breakdown_bytes,
    protected_pipeline_latency_s,
    snapshot_tx_bytes_at_completed_token,
)
from aerokv.specs import (
    ExecutionLayout,
    LayerInterval,
    LogicalRing,
    ModelSpec,
    ProtectionPlan,
    ProtectionRuntime,
    SystemSpec,
    UAVSpec,
)


def make_tiny_case():
    model = ModelSpec(
        num_layers=6,
        hidden_size=4,
        num_heads=1,
        n_est=20,
        bytes_per_value=2,
        model_params_billion=0.000000006,
    )
    uavs = tuple(
        UAVSpec(
            uav_id=i,
            memory_budget_bytes=1_000_000.0,
            initial_energy_j=1_000.0,
            flight_power_w=10.0,
            inference_power_w=5.0,
            per_layer_latency_s=0.1 * (i + 1),
            link_bps=100.0,
            tx_power_w=2.0,
        )
        for i in range(3)
    )
    system = SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=3.0,
        storage_load_bps=1_000.0,
        snapshot_period_candidates=(4,),
    )
    layout = ExecutionLayout(
        {
            0: LayerInterval(0, 2),
            1: LayerInterval(2, 4),
            2: LayerInterval(4, 6),
        }
    )
    ring = LogicalRing((0, 1, 2))
    plan = ProtectionPlan(
        method="AeroKV",
        layout=layout,
        ring=ring,
        head_overlap_depth={0: 1, 1: 1, 2: 1},
        snapshot_period={0: 4, 1: 4, 2: 4},
    )
    runtime = ProtectionRuntime(
        latest_snapshot_token={0: 8, 1: 8, 2: 8},
        activation_buffer_tokens={0: 2, 1: 2, 2: 2},
    )
    return system, plan, runtime


def test_memory_breakdown_for_ring_owner():
    system, plan, runtime = make_tiny_case()
    br = memory_breakdown_bytes(system, plan, uav_id=0, token=10, runtime=runtime)

    # hidden=4, bytes=2 -> KV bytes/layer/token = 16, activation bytes = 8
    # model weight bytes/layer = 2
    assert br.native_bytes == 2 * (2 + 10 * 16)
    assert br.live_overlap_bytes == 1 * (2 + 10 * 16)  # UAV0 stores source UAV1's head
    assert br.snapshot_bytes == 1 * 8 * 16             # UAV0 stores source UAV2's tail snapshot
    assert br.activation_buffer_bytes == 0.0         # activation buffer excluded from paper-level memory
    assert br.total_bytes == 614


def test_protected_pipeline_latency_hides_live_overlap_compute():
    system, plan, _runtime = make_tiny_case()
    br = protected_pipeline_latency_s(system, plan)
    assert math.isclose(br.stage_latency_by_uav[0], 2 * 0.1)
    assert math.isclose(br.stage_latency_by_uav[1], 2 * 0.2)
    assert math.isclose(br.stage_latency_by_uav[2], 2 * 0.3)
    assert math.isclose(br.pipeline_latency_s, 1.2)


def test_snapshot_tx_bytes_exact_and_average():
    system, plan, _runtime = make_tiny_case()
    assert snapshot_tx_bytes_at_completed_token(system, plan, source_uav=0, token=3) == 0
    assert snapshot_tx_bytes_at_completed_token(system, plan, source_uav=0, token=4) == 1 * 4 * 16
    assert average_snapshot_tx_bytes_per_token(system, plan, source_uav=0) == 16
    assert math.isclose(average_snapshot_tx_energy_per_token_j(system, plan, source_uav=0), 2.0 * 16 * 8 / 100.0)

if __name__ == "__main__":
    test_memory_breakdown_for_ring_owner()
    test_protected_pipeline_latency_hides_live_overlap_compute()
    test_snapshot_tx_bytes_exact_and_average()
    print("accounting tests passed")
