import math
from pathlib import Path

from aerokv.specs import ExecutionLayout, LayerInterval, LogicalRing, ModelSpec, SystemSpec, UAVSpec
from aerokv.simulation.events import FailureEvent, format_failure_history, generate_poisson_failure_events
from aerokv.protection_state import runtime_after_completed_token
from aerokv.recovery import compute_recovery
from aerokv.simulation.engine import simulate_fixed_plan_with_recovery, write_recovery_output
from aerokv.simulation.engine import make_fixed_protection_plan


def make_tiny_system():
    model = ModelSpec(
        num_layers=6,
        hidden_size=4,
        num_heads=1,
        n_est=6,
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
    layout = ExecutionLayout({0: LayerInterval(0, 2), 1: LayerInterval(2, 4), 2: LayerInterval(4, 6)})
    ring = LogicalRing((0, 1, 2))
    plan = make_fixed_protection_plan(system, layout, ring, method="AeroKV", overlap_depth=1, snapshot_period=2)
    return system, plan


def test_failure_history_format():
    events = (FailureEvent(356, 7), FailureEvent(2552, 10))
    assert format_failure_history(events) == "[token 356: uav 7, token 2552: uav 10]"


def test_poisson_failure_generation_is_sorted_and_unique():
    system, _ = make_tiny_case()
    events = generate_poisson_failure_events(system, expected_failures_per_task=2.5, seed=1, max_token=6)
    assert all(1 <= e.token <= 6 for e in events)
    assert [e.token for e in events] == sorted(e.token for e in events)
    assert len({e.failed_uav for e in events}) == len(events)


def test_compute_recovery_from_snapshot_runtime():
    system, plan = make_tiny_case()
    runtime = runtime_after_completed_token(system, plan, token=3)
    result = compute_recovery(
        system,
        plan,
        runtime,
        FailureEvent(token=3, failed_uav=1),
        alive_uavs_before_failure={0, 1, 2},
    )
    assert result.valid
    assert result.live_owner_uav == 0
    assert result.snapshot_owner_uav == 2
    assert result.live_head_layers == 1
    assert result.snapshot_tail_layers == 1
    assert result.latest_snapshot_token == 2
    assert result.replay_tokens == 1
    # load = 1 layer * 2 bytes/layer * 8 / 1000 bps = 0.016s; replay = 1 * 1 * 0.3 = 0.3s
    assert math.isclose(result.recovery_latency_s, 0.316)
    assert math.isclose(result.replay_compute_energy_by_uav_j[2], 1.5)
    assert result.recovered_exec_layers_by_uav == {0: 1, 2: 1}


def test_simulate_with_explicit_failure_and_complete_step_log(tmp_path: Path):
    system, plan = make_tiny_case()
    out = simulate_fixed_plan_with_recovery(
        system,
        plan,
        run_id="tiny-recovery",
        failure_events=(FailureEvent(token=3, failed_uav=1),),
        max_tokens=6,
        print_progress=False,
    )

    assert len(out.step_log) == 7  # token 0..6
    assert len(out.token_trace) == 7
    assert out.summary.num_failures == 1
    assert out.summary.failure_trace == "[token 3: uav 1]"
    assert math.isclose(out.summary.recovery_latency_s, 0.316)
    assert out.token_trace[3].phase == "post_failure"
    assert out.token_trace[3].failed_uav == 1
    assert out.token_trace[3].num_failures == 1
    assert out.token_trace[3].failure_history == "[token 3: uav 1]"
    assert out.token_trace[3].time_s > out.token_trace[2].time_s

    rows = {(row.token, row.uav_id): row for row in out.uav_trace}
    assert rows[(3, 1)].uav_status == "failed"
    assert rows[(3, 0)].recovered_exec_layers == 1
    assert rows[(3, 2)].recovered_exec_layers == 1

    write_recovery_output(out, tmp_path)
    for name in ("summary.csv", "token_trace.csv", "uav_trace.csv", "step_log.csv"):
        assert (tmp_path / name).exists()


if __name__ == "__main__":
    test_failure_history_format()
    test_poisson_failure_generation_is_sorted_and_unique()
    test_compute_recovery_from_snapshot_runtime()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_simulate_with_explicit_failure_and_complete_step_log(Path(d))
    print("recovery tests passed")
