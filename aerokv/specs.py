"""Core data contracts for the rewritten AeroKV simulator.

This module contains no planner, no simulator loop, and no plotting.
It defines the static system specification, layer layout, logical ring, protection
plan, and mutable runtime state needed by the later single-threaded standard
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ModelSpec:
    """Static model/workload specification.

    kv_bytes_per_token_layer_override is used when the paper specifies an
    empirical KV footprint directly (for Qwen-VL-32B, 4 KB per token per
    layer), instead of deriving it from hidden_size.
    """

    num_layers: int
    hidden_size: int
    num_heads: int
    n_est: int
    bytes_per_value: int
    model_params_billion: float
    kv_bytes_per_token_layer_override: int | None = None

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.n_est <= 0:
            raise ValueError("n_est must be positive")
        if self.bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be positive")
        if self.model_params_billion <= 0:
            raise ValueError("model_params_billion must be positive")

    @property
    def kv_bytes_per_token_layer(self) -> int:
        if self.kv_bytes_per_token_layer_override is not None:
            return self.kv_bytes_per_token_layer_override
        # K and V, each hidden_size values.
        return 2 * self.hidden_size * self.bytes_per_value

    @property
    def activation_bytes(self) -> int:
        return self.hidden_size * self.bytes_per_value

    @property
    def weight_bytes_per_layer(self) -> float:
        total_weight_bytes = self.model_params_billion * 1e9 * self.bytes_per_value
        return total_weight_bytes / self.num_layers


@dataclass(frozen=True)
class UAVSpec:
    """Static per-UAV resource and performance specification.

    No RX-side energy is modeled. Communication energy is TX-only.
    """

    uav_id: int
    memory_budget_bytes: float
    initial_energy_j: float
    flight_power_w: float
    inference_power_w: float
    per_layer_latency_s: float
    link_bps: float
    tx_power_w: float

    def __post_init__(self) -> None:
        if self.uav_id < 0:
            raise ValueError("uav_id must be non-negative")
        for name in (
            "memory_budget_bytes",
            "initial_energy_j",
            "flight_power_w",
            "inference_power_w",
            "per_layer_latency_s",
            "link_bps",
            "tx_power_w",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.link_bps == 0:
            raise ValueError("link_bps must be positive")


@dataclass(frozen=True)
class SystemSpec:
    """Complete static scenario for one standard single-threaded run."""

    model: ModelSpec
    uavs: tuple[UAVSpec, ...]
    tau_recover_max_s: float
    storage_load_bps: float
    snapshot_period_candidates: tuple[int, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.uavs) == 0:
            raise ValueError("at least one UAV is required")
        ids = [u.uav_id for u in self.uavs]
        if len(ids) != len(set(ids)):
            raise ValueError("uav_id values must be unique")
        if self.tau_recover_max_s < 0:
            raise ValueError("tau_recover_max_s must be non-negative")
        if self.storage_load_bps <= 0:
            raise ValueError("storage_load_bps must be positive")
        if any(p <= 0 for p in self.snapshot_period_candidates):
            raise ValueError("snapshot periods must be positive")

    @property
    def num_uavs(self) -> int:
        return len(self.uavs)

    def uav(self, uav_id: int) -> UAVSpec:
        for u in self.uavs:
            if u.uav_id == uav_id:
                return u
        raise KeyError(f"unknown uav_id {uav_id}")


@dataclass(frozen=True, order=True)
class LayerInterval:
    """Half-open layer interval [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("layer interval start must be non-negative")
        if self.end < self.start:
            raise ValueError("layer interval end must be >= start")

    @property
    def width(self) -> int:
        return self.end - self.start

    def is_empty(self) -> bool:
        return self.width == 0

    def clamp(self, start: int, end: int) -> "LayerInterval":
        return LayerInterval(max(self.start, start), min(self.end, end))


@dataclass(frozen=True)
class ExecutionLayout:
    """Layer ownership/execution layout: UAV id -> layer interval."""

    intervals: Mapping[int, LayerInterval]

    def executing_uavs(self) -> tuple[int, ...]:
        return tuple(u for u, interval in self.intervals.items() if interval.width > 0)

    def interval(self, uav_id: int) -> LayerInterval:
        return self.intervals[uav_id]

    def validate_exact_cover(self, num_layers: int) -> None:
        intervals = sorted((iv.start, iv.end) for iv in self.intervals.values() if iv.width > 0)
        cursor = 0
        for start, end in intervals:
            if start != cursor:
                raise ValueError(f"layout gap or overlap at layer {cursor}: next interval starts at {start}")
            cursor = end
        if cursor != num_layers:
            raise ValueError(f"layout covers [0, {cursor}), expected [0, {num_layers})")


@dataclass(frozen=True)
class LogicalRing:
    """Logical resilience ring over UAV ids."""

    uav_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.uav_ids) < 2:
            raise ValueError("logical ring requires at least two UAVs")
        if len(self.uav_ids) != len(set(self.uav_ids)):
            raise ValueError("logical ring contains duplicate UAV ids")

    def pred(self, uav_id: int) -> int:
        idx = self.uav_ids.index(uav_id)
        return self.uav_ids[(idx - 1) % len(self.uav_ids)]

    def succ(self, uav_id: int) -> int:
        idx = self.uav_ids.index(uav_id)
        return self.uav_ids[(idx + 1) % len(self.uav_ids)]

    def remove(self, failed_uav: int) -> "LogicalRing":
        remaining = tuple(u for u in self.uav_ids if u != failed_uav)
        return LogicalRing(remaining)


@dataclass(frozen=True)
class ProtectionPlan:
    """Static protection plan for a given execution layout.

    For source UAV i:
      - head_overlap_depth[i] is the number of leading layers of i's shard
        live-overlapped on pred(i).
      - snapshot_period[i] is the Boundary Snapshot period for the remaining
        tail layers sent to succ(i). None means no snapshot for that source.
    """

    method: str
    layout: ExecutionLayout
    ring: LogicalRing
    head_overlap_depth: Mapping[int, int]
    snapshot_period: Mapping[int, int | None]
    full_mirror: bool = False

    def validate_against(self, system: SystemSpec) -> None:
        self.layout.validate_exact_cover(system.model.num_layers)
        expected = set(self.layout.intervals.keys())
        if set(self.head_overlap_depth.keys()) != expected:
            raise ValueError("head_overlap_depth keys must match layout UAV ids")
        if set(self.snapshot_period.keys()) != expected:
            raise ValueError("snapshot_period keys must match layout UAV ids")
        for uav_id, k in self.head_overlap_depth.items():
            width = self.layout.interval(uav_id).width
            if k < 0 or k > width:
                raise ValueError(f"invalid overlap depth {k} for UAV {uav_id} with shard width {width}")
        for uav_id, period in self.snapshot_period.items():
            if period is not None and period <= 0:
                raise ValueError(f"snapshot period for UAV {uav_id} must be positive or None")


@dataclass(frozen=True)
class FailureCase:
    token: int
    failed_uav: int

    def validate_against(self, system: SystemSpec) -> None:
        if not (0 <= self.token <= system.model.n_est):
            raise ValueError("failure token must be within [0, n_est]")
        system.uav(self.failed_uav)


@dataclass
class ProtectionRuntime:
    """Compact runtime protection state.

    This is not a KV-segment debug table. It stores only the per-source freshness
    needed to compute memory, recovery latency, and feasibility.
    """

    latest_snapshot_token: dict[int, int] = field(default_factory=dict)
    activation_buffer_tokens: dict[int, int] = field(default_factory=dict)


@dataclass
class DroneRuntime:
    uav_id: int
    status: str
    energy_j: float
    memory_bytes: float = 0.0
    cumulative_flight_energy_j: float = 0.0
    cumulative_compute_energy_j: float = 0.0
    cumulative_tx_energy_j: float = 0.0

    def is_alive(self) -> bool:
        return self.status != "failed"


@dataclass
class RuntimeState:
    """Mutable state for a single standard run.

    ``execution_assignment`` is the runtime execution assignment.  Initially it
    matches ``layout`` exactly.  After failure recovery, a UAV may execute
    multiple non-contiguous layer intervals, so this is kept separate from the
    static ``ExecutionLayout`` used by the protection plan.
    """

    token: int
    time_s: float
    phase: str
    system: SystemSpec
    layout: ExecutionLayout
    ring: LogicalRing
    protection_plan: ProtectionPlan
    protection_runtime: ProtectionRuntime
    drones: dict[int, DroneRuntime]
    failed_uav: int | None = None
    execution_assignment: dict[int, tuple[LayerInterval, ...]] = field(default_factory=dict)
    failure_history: list[FailureCase] = field(default_factory=list)

    def alive_uavs(self) -> tuple[int, ...]:
        return tuple(u for u, state in self.drones.items() if state.is_alive())

    def min_alive_energy_j(self) -> float:
        energies = [d.energy_j for d in self.drones.values() if d.is_alive()]
        return min(energies) if energies else 0.0

    def failure_history_string(self) -> str:
        if not self.failure_history:
            return "[]"
        parts = [f"token {f.token}: uav {f.failed_uav}" for f in self.failure_history]
        return "[" + ", ".join(parts) + "]"
