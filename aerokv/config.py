"""Central configuration for the paper-aligned AeroKV experiments."""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024 ** 3
KB = 1024
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = DEFAULT_SEED

    # Conference draft setup: Qwen-VL-32B on a 16-UAV swarm.
    num_uavs: int = 16
    n_est: int = 8192
    num_layers: int = 64
    hidden_size: int = 12288
    num_heads: int = 96
    bytes_per_value: int = 2
    model_params_billion: float = 32.0
    kv_bytes_per_token_layer: int = 4 * KB

    # Default homogeneous resources.
    memory_budget_gb: float = 12.0
    energy_budget_kj_range: tuple[float, float] = (1800.0, 3600.0)

    # Optional heterogeneous profiles.
    uav_memory_budget_gb: tuple[float, ...] | None = None
    uav_compute_latency_multipliers: tuple[float, ...] | None = None
    uav_link_mbps: tuple[float, ...] | None = None

    # Deterministic power/latency model. These are max-power hardware constants;
    # runtime inference power and per-layer latency are derived from residual energy.
    flight_power_w: float = 80.0
    inference_power_max_w: float = 35.0
    inference_power_min_fraction: float = 0.40
    per_layer_latency_min_ms: float = 8.0
    power_latency_alpha: float = 0.70
    tx_power_w: float = 2.5
    link_rate_mbps_range: tuple[float, float] = (15.0, 30.0)

    # Recovery and failure settings.
    tau_recover_max_s: float = 3.0
    storage_load_gbps: float = 16.0
    snapshot_period_candidates: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    expected_failures_per_task: float = 2.5

    @property
    def memory_budget_bytes(self) -> float:
        return self.memory_budget_gb * GB

    @property
    def weight_bytes_per_layer(self) -> float:
        return self.model_params_billion * 1e9 * self.bytes_per_value / self.num_layers

    @property
    def storage_load_bps(self) -> float:
        return self.storage_load_gbps * 1e9


def make_default_config(seed: int = DEFAULT_SEED) -> ExperimentConfig:
    return ExperimentConfig(seed=seed)


def make_heterogeneous_config(seed: int = DEFAULT_SEED) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        num_uavs=16,
        num_layers=64,
        memory_budget_gb=12.0,
        uav_compute_latency_multipliers=(
            0.75, 0.90, 1.20, 1.00,
            0.70, 1.35, 1.10, 0.85,
            1.50, 0.80, 1.25, 1.05,
            0.65, 1.40, 0.95, 1.15,
        ),
        uav_link_mbps=(
            30.0, 24.0, 18.0, 27.0,
            22.0, 15.0, 20.0, 28.0,
            16.0, 30.0, 19.0, 25.0,
            29.0, 17.0, 26.0, 21.0,
        ),
        uav_memory_budget_gb=(
            12.0, 12.0, 10.0, 12.0,
            16.0, 10.0, 12.0, 14.0,
            10.0, 16.0, 12.0, 12.0,
            16.0, 10.0, 14.0, 12.0,
        ),
    )
