"""Poisson failure process for the standard AeroKV simulator.

Failures are generated in token space.  A failure at token t means that the UAV
fails immediately after token t has completed and before token t+1 starts.
The default expectation is 2.5 failures per task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import SystemSpec


@dataclass(frozen=True)
class FailureEvent:
    token: int
    failed_uav: int

    def validate_against(self, system: SystemSpec) -> None:
        if not (1 <= self.token <= system.model.n_est):
            raise ValueError(f"failure token must be within [1, {system.model.n_est}]")
        system.uav(self.failed_uav)


def format_failure_history(events: tuple[FailureEvent, ...] | list[FailureEvent]) -> str:
    if not events:
        return "[]"
    return "[" + ", ".join(f"token {e.token}: uav {e.failed_uav}" for e in events) + "]"


def generate_poisson_failure_events(
    system: SystemSpec,
    *,
    expected_failures_per_task: float = 2.5,
    seed: int | None = None,
    max_token: int | None = None,
) -> tuple[FailureEvent, ...]:
    """Sample a homogeneous Poisson failure process over the token horizon.

    ``expected_failures_per_task`` is the expected number of failures over
    ``system.model.n_est`` generated tokens.  Inter-arrival distances are
    exponential in token units.  Each failure is assigned uniformly among UAVs
    that have not already failed, so the same UAV is not failed twice.
    """

    if expected_failures_per_task < 0:
        raise ValueError("expected_failures_per_task must be non-negative")
    if max_token is None:
        max_token = system.model.n_est
    if not (1 <= max_token <= system.model.n_est):
        raise ValueError(f"max_token must be within [1, {system.model.n_est}]")
    if expected_failures_per_task == 0:
        return ()

    rng = np.random.default_rng(system.seed + 1701 if seed is None else seed)
    rate_per_token = expected_failures_per_task / float(system.model.n_est)
    alive = [u.uav_id for u in system.uavs]
    events: list[FailureEvent] = []
    t = 0.0

    while alive:
        t += float(rng.exponential(1.0 / rate_per_token))
        token = int(math.ceil(t))
        if events and token <= events[-1].token:
            token = events[-1].token + 1
        if token > max_token:
            break

        idx = int(rng.integers(0, len(alive)))
        failed = alive.pop(idx)
        events.append(FailureEvent(token=token, failed_uav=failed))

    return tuple(events)


# Lower-level helpers kept for direct token-schedule tests if needed.
def sample_failure_event_tokens(
    *,
    n_est: int,
    expected_failures_per_task: float = 2.5,
    seed: int = 2026,
) -> tuple[int, ...]:
    if n_est <= 0:
        raise ValueError("n_est must be positive")
    if expected_failures_per_task < 0:
        raise ValueError("expected_failures_per_task must be non-negative")
    if expected_failures_per_task == 0:
        return ()
    rng = np.random.default_rng(seed)
    rate = expected_failures_per_task / n_est
    t = 0.0
    events: list[int] = []
    while True:
        t += float(rng.exponential(1.0 / rate))
        token = int(math.ceil(t))
        if token > n_est:
            break
        events.append(max(1, token))
    return tuple(events)


def failure_events_by_token(tokens: tuple[int, ...]) -> dict[int, int]:
    out: dict[int, int] = {}
    for token in tokens:
        out[token] = out.get(token, 0) + 1
    return out
