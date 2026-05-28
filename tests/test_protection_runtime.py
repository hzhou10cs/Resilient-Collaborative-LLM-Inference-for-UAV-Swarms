import math

from aerokv2.accounting import memory_breakdown_bytes
from aerokv2.core import (
    ExecutionLayout,
    LayerInterval,
    LogicalRing,
    ModelSpec,
    ProtectionPlan,
    SystemSpec,
    UAVSpec,
)
from aerokv2.protection_runtime import (
    owner_protection_view,
    runtime_after_completed_token,
    source_protection_view,
    update_protection_runtime_at_completed_token,
    validate_runtime_consistency,
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
    return system, plan


def test_runtime_after_completed_token():
    system, plan = make_tiny_case()

    runtime = runtime_after_completed_token(system, plan, token=3)
    assert runtime.latest_snapshot_token == {0: 0, 1: 0, 2: 0}
    assert runtime.activation_buffer_tokens == {0: 3, 1: 3, 2: 3}
    validate_runtime_consistency(system, plan, 3, runtime)

    runtime = runtime_after_completed_token(system, plan, token=4)
    assert runtime.latest_snapshot_token == {0: 4, 1: 4, 2: 4}
    assert runtime.activation_buffer_tokens == {0: 0, 1: 0, 2: 0}
    validate_runtime_consistency(system, plan, 4, runtime)

    runtime = runtime_after_completed_token(system, plan, token=5)
    assert runtime.latest_snapshot_token == {0: 4, 1: 4, 2: 4}
    assert runtime.activation_buffer_tokens == {0: 1, 1: 1, 2: 1}
    validate_runtime_consistency(system, plan, 5, runtime)


def test_snapshot_tx_payload_at_update_token():
    system, plan = make_tiny_case()
    kv = system.model.kv_bytes_per_token_layer

    upd = update_protection_runtime_at_completed_token(system, plan, token=3)
    assert upd.total_snapshot_tx_bytes == 0
    assert upd.snapshot_sources == ()

    upd = update_protection_runtime_at_completed_token(system, plan, token=4)
    # Each source has one tail snapshot layer and emits period=4 tokens of KV.
    assert upd.snapshot_tx_bytes_by_source == {0: 4 * kv, 1: 4 * kv, 2: 4 * kv}
    assert upd.total_snapshot_tx_bytes == 3 * 4 * kv
    assert upd.snapshot_sources == (0, 1, 2)


def test_holder_direction_view():
    system, plan = make_tiny_case()
    runtime = runtime_after_completed_token(system, plan, token=5)

    view = owner_protection_view(system, plan, owner_uav=1, token=5, runtime=runtime)
    # Holder UAV 1 stores/computes succ(1)=2's head overlap.
    assert view.live_overlap_source_uav == 2
    assert view.live_overlap_layers == 1
    # Holder UAV 1 stores pred(1)=0's tail snapshot.
    assert view.snapshot_source_uav == 0
    assert view.snapshot_layers == 1
    assert view.latest_snapshot_token == 4
    assert view.snapshot_staleness_tokens == 1
    assert view.activation_buffer_tokens == 1


def test_source_direction_view():
    system, plan = make_tiny_case()
    runtime = runtime_after_completed_token(system, plan, token=4)

    view = source_protection_view(system, plan, source_uav=0, token=4, runtime=runtime)
    # Source UAV 0 sends head overlap to pred(0)=2 and tail snapshot to succ(0)=1.
    assert view.live_overlap_owner_uav == 2
    assert view.snapshot_owner_uav == 1
    assert view.head_overlap_layers == 1
    assert view.snapshot_tail_layers == 1
    assert view.snapshot_due is True
    assert view.latest_snapshot_token == 4
    assert view.snapshot_staleness_tokens == 0
    assert view.activation_buffer_tokens == 0


def test_runtime_feeds_memory_accounting():
    system, plan = make_tiny_case()
    runtime = runtime_after_completed_token(system, plan, token=5)
    br = memory_breakdown_bytes(system, plan, uav_id=1, token=5, runtime=runtime)

    # hidden=4, bytes=2 -> KV bytes/layer/token = 16, activation bytes = 8
    # model weight bytes/layer = 2
    assert br.native_bytes == 2 * (2 + 5 * 16)
    assert br.live_overlap_bytes == 1 * (2 + 5 * 16)  # UAV1 stores source UAV2's head
    assert br.snapshot_bytes == 1 * 4 * 16             # UAV1 stores source UAV0's tail snapshot
    assert br.activation_buffer_bytes == 1 * 8         # token 5 activation buffered since snapshot 4
    assert math.isclose(br.total_bytes, 318.0)


if __name__ == "__main__":
    test_runtime_after_completed_token()
    test_snapshot_tx_payload_at_update_token()
    test_holder_direction_view()
    test_source_direction_view()
    test_runtime_feeds_memory_accounting()
    print("protection runtime tests passed")
