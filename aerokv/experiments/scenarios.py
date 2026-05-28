"""Standard scenario construction for the rewritten AeroKV simulator."""

from __future__ import annotations

import numpy as np

from ..specs import ExecutionLayout, LayerInterval, LogicalRing, ModelSpec, SystemSpec, UAVSpec

GB = 1024**3


def contiguous_partition(num_layers: int, num_parts: int) -> tuple[LayerInterval, ...]:
    if num_layers <= 0 or num_parts <= 0:
        raise ValueError("num_layers and num_parts must be positive")
    base = num_layers // num_parts
    extra = num_layers % num_parts
    out: list[LayerInterval] = []
    start = 0
    for idx in range(num_parts):
        width = base + (1 if idx < extra else 0)
        out.append(LayerInterval(start, start + width))
        start += width
    return tuple(out)


def make_standard_system(seed: int = 2026, config=None) -> SystemSpec:
    """Build the paper-aligned standard scenario from ``ExperimentConfig``."""

    from ..config import ExperimentConfig

    cfg = config if config is not None else ExperimentConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    model = ModelSpec(
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        n_est=cfg.n_est,
        bytes_per_value=cfg.bytes_per_value,
        model_params_billion=cfg.model_params_billion,
        kv_bytes_per_token_layer_override=cfg.kv_bytes_per_token_layer,
    )
    uavs = tuple(
        UAVSpec(
            uav_id=i,
            memory_budget_bytes=cfg.memory_budget_bytes,
            initial_energy_j=float(rng.uniform(*cfg.energy_budget_kj_range) * 1000.0),
            flight_power_w=float(cfg.flight_power_w),
            inference_power_w=float(cfg.inference_power_max_w),
            per_layer_latency_s=float(cfg.per_layer_latency_min_ms / 1000.0),
            link_bps=float(rng.uniform(*cfg.link_rate_mbps_range) * 1e6),
            tx_power_w=cfg.tx_power_w,
        )
        for i in range(cfg.num_uavs)
    )
    return SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=cfg.tau_recover_max_s,
        storage_load_bps=cfg.storage_load_bps,
        snapshot_period_candidates=cfg.snapshot_period_candidates,
        seed=cfg.seed if seed is None else seed,
    )

def make_initial_layout(system: SystemSpec) -> ExecutionLayout:
    intervals = contiguous_partition(system.model.num_layers, system.num_uavs)
    layout = ExecutionLayout({uav.uav_id: intervals[pos] for pos, uav in enumerate(system.uavs)})
    layout.validate_exact_cover(system.model.num_layers)
    return layout


def make_initial_ring(system: SystemSpec) -> LogicalRing:
    return LogicalRing(tuple(u.uav_id for u in system.uavs))
