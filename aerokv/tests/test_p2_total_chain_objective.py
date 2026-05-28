"""Regression test: P2 optimizes total chain latency, not max stage latency."""

import math

from aerokv.optimizers.p2_reconfiguration import p2_stage_latency_s, solve_p2_reconfiguration
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


def _toy_system() -> SystemSpec:
    model = ModelSpec(
        num_layers=10,
        hidden_size=8,
        num_heads=1,
        n_est=128,
        bytes_per_value=2,
        model_params_billion=0.000001,
        kv_bytes_per_token_layer_override=1,
    )
    uavs = (
        UAVSpec(0, 10**12, 10**9, 0.0, 1.0, 1.0, 10**9, 0.0),
        UAVSpec(1, 10**12, 10**9, 0.0, 1.0, 7.0, 10**9, 0.0),
        UAVSpec(2, 10**12, 10**9, 0.0, 1.0, 2.0, 10**9, 0.0),
    )
    return SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=1000.0,
        storage_load_bps=10**9,
        snapshot_period_candidates=(1,),
        seed=1,
    )


def test_p2_minimizes_total_chain_latency_not_max_stage_latency():
    system = _toy_system()
    layout = ExecutionLayout(
        {
            0: LayerInterval(0, 4),
            1: LayerInterval(4, 7),
            2: LayerInterval(7, 10),
        }
    )
    ring = LogicalRing((0, 1, 2))
    plan = ProtectionPlan(
        method="AeroKV",
        layout=layout,
        ring=ring,
        head_overlap_depth={0: 0, 1: 0, 2: 0},
        snapshot_period={0: None, 1: None, 2: None},
    )
    runtime = ProtectionRuntime()

    # Make every layer state-available to every survivor so the objective alone
    # selects the partition. With latencies [1, 7, 2], total-chain minimization
    # gives as many layers as possible to UAV 0: widths [8, 1, 1].
    all_layers = (LayerInterval(0, 10),)
    result = solve_p2_reconfiguration(
        system,
        plan,
        runtime,
        token=0,
        alive_uavs={0, 1, 2},
        recovered_intervals_by_uav={0: all_layers, 1: all_layers, 2: all_layers},
    )

    assert result.valid, result.invalid_reason
    assert result.layout is not None
    assert result.layout.interval(0) == LayerInterval(0, 8)
    assert result.layout.interval(1) == LayerInterval(8, 9)
    assert result.layout.interval(2) == LayerInterval(9, 10)
    assert result.compute_latencies_s == (8.0, 7.0, 2.0)
    assert result.activation_forward_latencies_s[0] > 0.0
    assert result.activation_forward_latencies_s[1] > 0.0
    assert result.activation_forward_latencies_s[2] == 0.0
    assert math.isclose(result.stage_latencies_s[0], 8.0 + result.activation_forward_latencies_s[0])
    assert math.isclose(result.stage_latencies_s[1], 7.0 + result.activation_forward_latencies_s[1])
    assert math.isclose(result.stage_latencies_s[2], 2.0)
    assert math.isclose(result.total_chain_latency_s, sum(result.stage_latencies_s))
    assert math.isclose(result.max_stage_latency_s, max(result.stage_latencies_s))

    uniform = ExecutionLayout(
        {
            0: LayerInterval(0, 4),
            1: LayerInterval(4, 7),
            2: LayerInterval(7, 10),
        }
    )
    active = (0, 1, 2)
    uniform_total = 0.0
    for idx, uav_id in enumerate(active):
        next_uav = active[idx + 1] if idx + 1 < len(active) else None
        stage, _, _, _ = p2_stage_latency_s(system, uav_id, uniform.interval(uav_id), next_uav)
        uniform_total += stage
    assert result.total_chain_latency_s < uniform_total
