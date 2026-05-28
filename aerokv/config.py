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

    # UAV resources.
    memory_budget_gb: float = 12.0
    energy_budget_kj_range: tuple[float, float] = (1800.0, 3600.0)
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
