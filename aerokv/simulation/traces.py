"""Trace row schemas for the standard AeroKV experiment.

The rewrite records only key per-token and per-UAV data. There is no event log,
no RX energy, no bottleneck flag, and no KV segment debug output by default.
Layer 4 adds a compact complete step log for progress/debug visibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TokenTraceRow:
    run_id: str
    method: str
    token: int
    time_s: float
    phase: str
    failed_uav: int | None
    num_alive_uavs: int
    pipeline_latency_s: float
    min_energy_j: float
    total_energy_j: float
    total_memory_bytes: float
    max_memory_bytes: float
    remaining_tokens: int
    system_expected_remaining_tokens: float
    cumulative_compute_energy_j: float
    cumulative_flight_energy_j: float
    cumulative_tx_energy_j: float
    num_failures: int = 0
    failure_history: str = "[]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UAVTraceRow:
    run_id: str
    method: str
    token: int
    time_s: float
    uav_id: int
    uav_status: str
    energy_j: float
    flight_energy_j: float
    compute_energy_j: float
    tx_energy_j: float
    memory_bytes: float
    native_layer_start: int | None
    native_layer_end: int | None
    exec_layer_start: int | None
    exec_layer_end: int | None
    num_exec_layers: int
    live_overlap_layers: int
    snapshot_layers: int
    latest_snapshot_token: int | None
    snapshot_staleness_tokens: int | None
    activation_buffer_tokens: int
    stage_latency_s: float
    expected_remaining_tokens: float
    recovered_exec_layers: int = 0
    inference_power_w: float = 0.0
    tx_power_w: float = 0.0
    overlap_power_w: float = 0.0
    snapshot_boundary_power_w: float = 0.0
    total_future_power_w: float = 0.0
    per_token_latency_s: float = 0.0
    token_rate: float = 0.0
    energy_per_token_j: float = 0.0
    predicted_remaining_tokens_i: float = 0.0
    activation_forward_energy_per_token_j: float = 0.0
    snapshot_boundary_energy_per_token_j: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepLogRow:
    """Compact complete per-token log.

    This is not an event log. It has one row per completed token and is meant
    for quick run inspection and reproducible progress output.
    """

    run_id: str
    token: int
    time_s: float
    phase: str
    num_alive_uavs: int
    avg_energy_used_j: float
    min_energy_j: float
    failure_history: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryRow:
    run_id: str
    method: str
    seed: int
    num_uavs: int
    num_layers: int
    n_est: int
    failed_uav: int | None
    failure_token: int | None
    deadline_s: float
    recovery_latency_s: float
    deadline_met: bool
    reconfiguration_latency_s: float
    remaining_completion_time_s: float
    mission_complete_s: float | None
    terminal_min_energy_j: float
    protection_compute_energy_j: float
    tx_energy_j: float
    reconfiguration_energy_j: float
    mission_success: bool
    invalid_reason: str | None = None
    num_failures: int = 0
    failure_trace: str = "[]"
    expected_failures_per_task: float | None = None
    total_recovery_latency_s: float = 0.0
    final_system_expected_remaining_tokens: float = 0.0
    p2_valid: bool | None = None
    p1_new_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
