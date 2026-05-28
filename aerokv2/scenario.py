"""Standard scenario construction for the rewritten AeroKV simulator."""

from __future__ import annotations

import numpy as np

from .core import ExecutionLayout, LayerInterval, LogicalRing, ModelSpec, SystemSpec, UAVSpec

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


def make_standard_system(seed: int = 2026) -> SystemSpec:
    """Mirror the current default experiment scale, without plotting fields."""

    rng = np.random.default_rng(seed)
    num_uavs = 12
    model = ModelSpec(
        num_layers=60,
        hidden_size=6656,
        num_heads=52,
        n_est=4096,
        bytes_per_value=2,
        model_params_billion=30.0,
    )
    uavs = tuple(
        UAVSpec(
            uav_id=i,
            memory_budget_bytes=24.0 * GB,
            initial_energy_j=float(rng.uniform(120.0, 180.0) * 1000.0),
            flight_power_w=float(rng.uniform(100.0, 160.0)),
            inference_power_w=float(rng.uniform(15.0, 35.0)),
            per_layer_latency_s=float(rng.uniform(8.0, 14.0) / 1000.0),
            link_bps=float(rng.uniform(200.0, 800.0) * 1e6),
            tx_power_w=2.5,
        )
        for i in range(num_uavs)
    )
    return SystemSpec(
        model=model,
        uavs=uavs,
        tau_recover_max_s=3.0,
        storage_load_bps=16.0 * 1e9,
        snapshot_period_candidates=(16, 32, 64, 128, 256, 512, 1024),
        seed=seed,
    )


def make_initial_layout(system: SystemSpec) -> ExecutionLayout:
    intervals = contiguous_partition(system.model.num_layers, system.num_uavs)
    layout = ExecutionLayout({uav.uav_id: intervals[pos] for pos, uav in enumerate(system.uavs)})
    layout.validate_exact_cover(system.model.num_layers)
    return layout


def make_initial_ring(system: SystemSpec) -> LogicalRing:
    return LogicalRing(tuple(u.uav_id for u in system.uavs))
