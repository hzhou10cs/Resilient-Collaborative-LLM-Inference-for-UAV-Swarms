# Resilient Collaborative LLM Inference for UAV Swarms

## Local changes compared with GitHub `main`

This local checkout differs from `origin/main`. The changes were made to debug advisor-raised result inconsistencies and to bring the simulator closer to the paper's stated P1/P2 model.

Main changes:

- P2 now optimizes total chain latency, not maximum stage latency. The paper defines post-failure runtime as the sum of surviving stage latencies, so `p2_reconfiguration.py` was changed from a stage-balancing objective to a total-latency objective. P2 diagnostics now also show availability evidence, selected layer intervals, compute latency, activation-forward latency, total chain latency, and max stage latency.
- `predicted_remaining_tokens` now uses one consistent formula for all methods: `min_i remaining_energy_j[i] / future_energy_per_token_j[i]`. This was changed because seed `100` showed SO and AeroKV with nearly identical energy and task time but a roughly 2x difference in predicted remaining tokens. The future-token estimate now uses the bottleneck UAV and includes flight, inference, live-overlap compute, snapshot/boundary TX, and activation forwarding.
- The simulator now records more final per-UAV diagnostics in `uav_trace.csv`, including inference power, TX power, overlap power, snapshot/boundary power, total future power, per-token latency, energy per future token, and each UAV's predicted remaining token capacity.
- AeroKV failure handling now records whether P2 and P1-new were valid, so failed recovery, failed P2, and failed P1-new states are easier to interpret consistently across methods.
- P1, P1-new, and P2 print more detailed debug logs. These logs explain why candidates were accepted or rejected, what layer state is available after failure, and why a post-failure layout was selected.
- Optional heterogeneous UAV profiles were added for debugging P2 behavior when UAVs have different compute speeds, link rates, and memory budgets.
- New audit scripts and tests were added, including a seed sweep for `[0, 1, 2, 3, 10, 42, 100, 2026]` and regression tests for the P2 total-chain objective and the remaining-token invariant.

Why these changes were made:

- The recovery-latency math already matched the paper, but P2 previously used a max-stage objective while the paper describes a sum-of-stage objective.
- Some experiment outputs were hard to explain because task time, remaining energy, and predicted remaining tokens were not transparent enough at the per-UAV level.
- The seed `100` SO/AeroKV result suggested that `predicted_remaining_tokens` was being computed with an inconsistent future-energy denominator.

Useful local files:

- `aerokv/experiments/remaining_tokens_audit.py`: runs the remaining-token seed sweep and writes final per-UAV diagnostics.
- `aerokv/experiments/p2_heterogeneous_compare.py`: compares selected P2, uniform layout, and no-reorganization behavior under heterogeneous UAV profiles.
- `aerokv/tests/test_p2_total_chain_objective.py`: verifies that P2 minimizes total chain latency.
- `aerokv/tests/test_expected_tokens.py`: verifies the bottleneck remaining-token formula and invariant.

This repository contains a single-threaded simulator for evaluating resilient collaborative LLM inference over a logical UAV ring.  The current implementation compares four methods:

- `NP`: no proactive protection; after a failure, the system performs hard recovery.
- `OO`: overlap-only protection; each UAV maintains live overlap for its successor.
- `SO`: snapshot-only protection; full-shard snapshot baseline with a fixed snapshot period.
- `AeroKV`: live head overlap plus boundary snapshot, with P1 provisioning before execution and P2/P1-new after failures.

The simulator is intended for reproducible experiments and debugging, not real-time deployment.

## Repository layout

```text
aerokv/
  config.py                         # Global experiment defaults.
  specs.py                          # Core dataclasses: model, UAV, layout, ring, runtime state.
  layout.py                         # Layout helpers, if present in your checkout.
  topology.py                       # LogicalRing export / topology compatibility layer.
  protection_state.py               # Runtime snapshot freshness and protection-state updates.
  accounting.py                     # Memory, energy, latency, snapshot, and runtime power formulas.
  recovery.py                       # Method-specific failure recovery accounting.
  baselines.py                      # NP / OO / SO / AeroKV initial plan construction.

  optimizers/
    p1_provisioning.py              # P1 provisioning: choose K and T_B using candidate search.
    p2_reconfiguration.py           # State-constrained P2 post-failure reconfiguration.
    p1_new.py                       # P1-new provisioning after P2 layout.

  simulation/
    events.py                       # Failure event representation and Poisson failure schedules.
    engine.py                       # Main single-threaded simulation loop.
    traces.py                       # CSV row schemas for summary/token/UAV/step logs.
    metrics.py                      # Experiment result aggregation helpers, if present.

  experiments/
    scenarios.py                    # Standard system construction.
    _common.py                      # Shared experiment utilities and result-table printing.
    exp1.py                         # Experiment 1 entry point.
    exp2.py                         # Experiment 2 entry point.
    exp3.py                         # Experiment 3 entry point.

  tests/
    test_memory_accounting.py
    test_recovery_latency.py
    test_p1_feasibility.py
    test_p2_state_constraints.py
    test_p2_min_one_layer.py
    test_ring_edges.py
    test_baselines.py
```

## Main modeling assumptions

### Energy and latency

The current runtime model uses battery-aware inference power and per-layer latency.  The high-level intended behavior is:

- When remaining battery is above 40%, inference runs at full power.
- Between 20% and 40%, inference power decays nonlinearly.
- At or below 20%, inference reaches its minimum power fraction.
- Per-layer latency increases as inference power decreases.

Flight power is fixed by the scenario configuration, currently intended to be `80 W` per UAV.  Communication energy is TX-only.  RX energy is not modeled.

### Pipeline latency

Live-overlap compute is hidden in the simplified bubble model.  It still consumes compute energy and memory, but it should not directly add to the main decode pipeline latency.

The simulation engine uses its configured pipeline-latency calculation for token-time advance.  When debugging latency behavior, inspect:

```text
aerokv/accounting.py
aerokv/simulation/engine.py
```

### Memory

Memory follows the paper-level model: native weights/KV, live-overlap weights/KV, and snapshot KV.  Activation-buffer memory is intentionally not counted in the main memory budget unless explicitly reintroduced.

### Recovery and reconfiguration

AeroKV failure path:

```text
failure
→ recovery
→ P2 state-constrained reconfiguration
→ P1-new protection rebuilding
→ continue execution
```

For simplification, P2/P1-new planner latency is modeled as zero and reconfiguration energy is currently treated as a fixed overhead, e.g. `200 J`, rather than detailed data-movement accounting.

NP / OO / SO use method-specific recovery.  After a failure, the surviving logical ring should be rewired so that predecessor/successor relationships skip failed UAVs.

### P2 minimum shard width

P2 now enforces a minimum of one layer per active surviving UAV.  Zero-width intervals are forbidden because they create invalid post-failure ring/protection semantics after P1-new.  If there are more active UAVs than model layers, P2 returns an invalid result rather than assigning empty shards.

## Key configuration settings

Most defaults are in:

```text
aerokv/config.py
aerokv/experiments/scenarios.py
```

Important fields to check before running experiments:

```python
num_uavs
num_layers
n_est
memory_budget_gb
initial_energy_j or energy budget setting
flight_power_w
link_rate_mbps_range
snapshot_period_candidates
expected_failures_per_task
tau_recover_max_s
```

Current recommended snapshot candidate set for AeroKV P1:

```python
snapshot_period_candidates = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
```

SO is usually configured as a full-shard snapshot baseline with fixed `T_B = 32`, unless an experiment intentionally sweeps it.

## Installing and running

From the repository root:

```bash
python -m pip install -U pytest
```

No GPU or LLM runtime is required; this is a metadata/energy/latency simulator.

Run tests:

```bash
PYTHONPATH=. pytest -q aerokv/tests
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='.'
pytest -q aerokv/tests
```

## Running experiments

### Experiment 1

```bash
PYTHONPATH=. python -m aerokv.experiments.exp1 --seed 2026 --output-dir outputs/exp1
```

Windows PowerShell:

```powershell
$env:PYTHONPATH='.'
python -m aerokv.experiments.exp1 --seed 2026 --output-dir outputs/exp1
```

Disable progress display:

```bash
PYTHONPATH=. python -m aerokv.experiments.exp1 --seed 2026 --output-dir outputs/exp1 --no-progress
```

Expected console output is a compact table like:

```text
Experiment 1 main results
method  avg_remaining_energy_j  task_time_cost_s  predicted_remaining_tokens
NP                         ...               ...                       ...
OO                         ...               ...                       ...
SO                         ...               ...                       ...
AeroKV                     ...               ...                       ...
```

### Experiment 2

```bash
PYTHONPATH=. python -m aerokv.experiments.exp2 --seed 2026 --output-dir outputs/exp2
```

Use `--no-progress` for quieter output.

### Experiment 3

```bash
PYTHONPATH=. python -m aerokv.experiments.exp3 --seed 2026 --output-dir outputs/exp3
```

Use `--no-progress` for quieter output.

## Output files

Each run writes CSVs under the selected output directory.  Depending on the experiment, files may be nested by method or setting.

Common files:

```text
summary.csv       # One-row or aggregated summary for the run.
token_trace.csv   # Per-token system state.
uav_trace.csv     # Per-token-per-UAV state.
step_log.csv      # Per-token concise log state used for progress/debugging.
```

Important columns:

```text
summary.csv:
  method
  mission_success
  invalid_reason
  task_time_cost_s or mission_complete_s
  terminal_min_energy_j
  final_system_expected_remaining_tokens
  recovery_latency_s
  reconfiguration_latency_s
  reconfiguration_energy_j
  failure_trace

token_trace.csv:
  token
  time_s
  phase
  num_alive_uavs
  pipeline_latency_s
  min_energy_j
  total_energy_j
  total_memory_bytes
  max_memory_bytes
  system_expected_remaining_tokens
  failure_history

uav_trace.csv:
  token
  uav_id
  uav_status
  energy_j
  memory_bytes
  num_exec_layers
  live_overlap_layers
  snapshot_layers
  latest_snapshot_token
  expected_remaining_tokens
  recovered_exec_layers
```

## Debugging guide

### If a method appears faster than AeroKV

Check whether it completed the whole task or failed early.  A failed method may have a smaller `task_time_cost_s` simply because it stopped before generating all tokens.  Inspect:

```text
summary.csv: mission_success, invalid_reason, failure_trace
token_trace.csv: final token, phase, remaining_tokens
```

### If AeroKV exits early

Look at:

```text
summary.csv: invalid_reason
token_trace.csv: last row phase/failure_history
uav_trace.csv: final num_exec_layers and energy_j
```

Common causes:

```text
recovery_failed
p2_reconfiguration_failed
p1_new_failed
memory_budget_exceeded
energy_depleted
```

### If 0-layer UAVs appear

This should no longer occur after this patch.  P2 enforces minimum one layer per active UAV.  If it reappears, inspect:

```text
aerokv/optimizers/p2_reconfiguration.py
```

and check whether another code path is constructing post-failure layouts outside P2.

### If baselines fail after multiple failures

Verify ring rewiring.  After UAV `i` fails, the surviving ring should remove `i`, so predecessor/successor skip the failed node.  Relevant files:

```text
aerokv/specs.py              # LogicalRing.remove
aerokv/simulation/engine.py  # When ring/layout/plan are updated after recovery
aerokv/recovery.py           # Method-specific recovery using current ring
```

### If memory looks too small

This is expected under the current paper parameters.  With `4 KB/token/layer` and `8192` tokens, one layer of full KV history is about `32 MB`; with roughly four layers per UAV, native KV is roughly `128 MB`, much smaller than model weights and a 12 GB memory budget.

## Development notes

- Keep planner logic out of accounting formulas.
- Keep experiment result printing in `experiments/_common.py`.
- Do not reintroduce `Ideal Full Mirror` unless its semantics and memory/energy accounting are explicitly defined.
- Do not count live-overlap compute as pipeline latency unless the bubble model is replaced by a more detailed scheduler.
- Do not allow P2 zero-width intervals in the standard experiments.
