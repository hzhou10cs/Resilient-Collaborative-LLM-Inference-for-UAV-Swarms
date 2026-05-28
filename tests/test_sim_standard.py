import csv
import math
from pathlib import Path

from aerokv2.core import ExecutionLayout, LayerInterval, LogicalRing, ModelSpec, SystemSpec, UAVSpec
from aerokv2.sim_standard import make_fixed_protection_plan, simulate_fixed_plan_prefailure, write_standard_output


def make_tiny_system():
    model = ModelSpec(
        num_layers=6,
        hidden_size=4,
        num_heads=1,
        n_est=5,
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
    return SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=3.0,
        storage_load_bps=1_000.0,
        snapshot_period_candidates=(2,),
        seed=7,
    )


def make_tiny_case():
    system = make_tiny_system()
    layout = ExecutionLayout(
        {
            0: LayerInterval(0, 2),
            1: LayerInterval(2, 4),
            2: LayerInterval(4, 6),
        }
    )
    ring = LogicalRing((0, 1, 2))
    plan = make_fixed_protection_plan(
        system,
        layout,
        ring,
        method="AeroKV",
        overlap_depth=1,
        snapshot_period=2,
    )
    return system, plan


def test_standard_pre_failure_row_counts_and_time():
    system, plan = make_tiny_case()
    result = simulate_fixed_plan_prefailure(system, plan, run_id="tiny")

    assert len(result.token_trace) == system.model.n_est + 1
    assert len(result.uav_trace) == (system.model.n_est + 1) * system.num_uavs
    assert result.token_trace[0].token == 0
    assert result.token_trace[0].time_s == 0.0
    assert result.token_trace[-1].token == 5
    assert math.isclose(result.token_trace[-1].time_s, 4.5)
    assert math.isclose(result.summary.mission_complete_s, 4.5)


def test_standard_pre_failure_energy_and_tx_bursts():
    system, plan = make_tiny_case()
    result = simulate_fixed_plan_prefailure(system, plan, run_id="tiny")

    token1 = result.token_trace[1]
    assert math.isclose(token1.cumulative_flight_energy_j, 27.0)
    assert math.isclose(token1.cumulative_compute_energy_j, 9.0)
    assert math.isclose(token1.cumulative_tx_energy_j, 0.0)

    token2 = result.token_trace[2]
    # Each of 3 sources sends one tail layer * period 2 * 16 bytes/layer/token.
    # TX energy per source = 2 W * 32 bytes * 8 / 100 bps = 5.12 J.
    assert math.isclose(token2.cumulative_tx_energy_j, 15.36)

    assert math.isclose(result.summary.tx_energy_j, 30.72)
    assert math.isclose(result.summary.terminal_min_energy_j, 922.26)


def test_standard_pre_failure_uav_trace_contains_holder_view():
    system, plan = make_tiny_case()
    result = simulate_fixed_plan_prefailure(system, plan, run_id="tiny")

    rows = {(row.token, row.uav_id): row for row in result.uav_trace}
    uav1_t2 = rows[(2, 1)]

    assert uav1_t2.native_layer_start == 2
    assert uav1_t2.native_layer_end == 4
    assert uav1_t2.exec_layer_start == 2
    assert uav1_t2.exec_layer_end == 4
    assert uav1_t2.num_exec_layers == 2
    assert uav1_t2.live_overlap_layers == 1       # holder 1 computes succ(1)=2's head
    assert uav1_t2.snapshot_layers == 1           # holder 1 stores pred(1)=0's tail
    assert uav1_t2.latest_snapshot_token == 2
    assert uav1_t2.snapshot_staleness_tokens == 0
    assert uav1_t2.activation_buffer_tokens == 0
    assert math.isclose(uav1_t2.stage_latency_s, 0.6)
    assert math.isclose(uav1_t2.memory_bytes, 134.0)

    uav1_t3 = rows[(3, 1)]
    assert uav1_t3.latest_snapshot_token == 2
    assert uav1_t3.snapshot_staleness_tokens == 1
    assert uav1_t3.activation_buffer_tokens == 1


def test_write_standard_outputs(tmp_path: Path):
    system, plan = make_tiny_case()
    result = simulate_fixed_plan_prefailure(system, plan, run_id="tiny")
    write_standard_output(result, tmp_path)

    for name in ("summary.csv", "token_trace.csv", "uav_trace.csv"):
        assert (tmp_path / name).exists()

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "tiny"
    assert rows[0]["method"] == "AeroKV"


if __name__ == "__main__":
    test_standard_pre_failure_row_counts_and_time()
    test_standard_pre_failure_energy_and_tx_bursts()
    test_standard_pre_failure_uav_trace_contains_holder_view()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_write_standard_outputs(Path(d))
    print("standard simulation tests passed")
