# AeroKV rewrite layer 5

This layer adds the final two planned components without changing the trace schema or adding threading:

- `aerokv2/p2_reconfiguration.py`
- `aerokv2/p1_provisioning.py`
- `aerokv2/p1_new.py`

It also extends `RecoveryResult` with `recovered_intervals_by_uav`, because P2 needs exact layer intervals rather than only recovered layer counts.

## P2 semantics

`solve_p2_reconfiguration(...)` is state-constrained. A surviving UAV may claim a layer only if that layer is available from one of these sources:

1. its own native shard;
2. the live-overlapped head of its ring successor;
3. the snapshot-recoverable tail of its ring predecessor;
4. intervals already materialized by the recovery step.

There is no unconstrained uniform fallback. If no exact state-constrained cover exists, the result is invalid and `layout` is `None`.

The objective is minimal maximum stage latency over feasible contiguous exact-cover layouts.

## P1 semantics

`solve_p1_provisioning(...)` performs deadline-guided candidate generation and bounded joint beam search. It returns only globally validated plans:

- memory feasible at `n_est`;
- worst-case recovery latency within `tau_recover_max_s`;
- objective is max-min pre-failure lifetime under protected steady-state energy.

This is not the old per-UAV independent greedy planner.

## P1-new semantics

`solve_p1_new(...)` re-solves P1 on the P2 layout and a surviving logical ring whose order follows the P2 active-UAV order.

## Current integration status

The modules are implemented and tested as standalone layers. `sim_recovery.py` remains the previous fixed-plan recovery simulator; it does not automatically call P2 or P1-new yet. That integration should be the next layer if desired.

## Tests

Run:

```bash
cd aerokv_rewrite_layer5
PYTHONPATH=. pytest -q
```

Expected result:

```text
20 passed
```
