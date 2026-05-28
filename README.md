# AeroKV Simulator

This repository contains a single-threaded AeroKV simulator for resilient collaborative LLM/VLM inference on UAV swarms. The implementation is organized around explicit system specifications, layer layouts, logical-ring protection state, accounting formulas, recovery, and optimization modules.

## Directory layout

```text
aerokv/
  config.py
  specs.py
  layout.py
  topology.py
  protection_state.py
  accounting.py
  recovery.py
  baselines.py

  optimizers/
    p1_provisioning.py
    p2_reconfiguration.py
    p1_new.py

  simulation/
    events.py
    engine.py
    traces.py
    metrics.py

  experiments/
    scenarios.py
    exp1.py
    exp2.py
    exp3.py

  tests/
    test_memory_accounting.py
    test_recovery_latency.py
    test_p1_feasibility.py
    test_p2_state_constraints.py
    test_ring_edges.py
    test_baselines.py
```

## Paper-aligned default parameters

The defaults in `aerokv/config.py` follow the latest conference draft experiment section:

```text
UAV count:                 16
Generated tokens/task:      8192
Base model:                 Qwen-VL-32B
Decoder layers:             64
Hidden size:                12288
KV footprint:               4 KB per token per layer
Inference memory budget:    12 GB per UAV
Initial energy:             uniform [120, 180] kJ
Flight-maintenance power:   uniform [100, 160] W
Inference power:            uniform [15, 35] W
TX power:                   2.5 W
Logical-ring link rate:     uniform [200, 800] Mbps
Recovery deadline:          3.0 s
Failure process:            token-space Poisson process
Expected failures/task:     2.5 by default
```

The simulator models TX-side communication energy only. It does not introduce RX energy, bottleneck flags, event-log output, or multiprocessing.

## Core conventions

Token convention:

```text
token = 0      initial state before generation
token = t > 0  state after completing t generated tokens
```

Failure convention:

```text
A failure at token t occurs after token t completes and before token t+1 starts.
```

AeroKV protection direction:

```text
Source UAV i:
  head overlap is stored/computed on pred(i)
  tail snapshot is stored on succ(i)

Holder UAV i:
  stores/computes succ(i)'s head overlap
  stores pred(i)'s tail snapshot
```

## Running experiments

Experiment 1: overall end-to-end performance under the same failure trace.

```bash
PYTHONPATH=. python -m aerokv.experiments.exp1 --seed 2026 --output-dir outputs/exp1
# or
PYTHONPATH=. python -m aerokv.experiments.run_fig1_lifetime --seed 2026
```

Experiment 2: recovery communication overhead comparison.

```bash
PYTHONPATH=. python -m aerokv.experiments.exp2 --seed 2026 --output-dir outputs/exp2
# or
PYTHONPATH=. python -m aerokv.experiments.run_fig2_tradeoff --seed 2026
```

Experiment 3: ablation study.

```bash
PYTHONPATH=. python -m aerokv.experiments.exp3 --seed 2026 --output-dir outputs/exp3
# or
PYTHONPATH=. python -m aerokv.experiments.run_fig3_reconfiguration --seed 2026
```

## Outputs

The recovery-enabled engine writes:

```text
summary.csv
 token_trace.csv
 uav_trace.csv
 step_log.csv
```

`step_log.csv` is complete per-token logging. Console progress, when enabled, prints one concise line every 20 tokens.

## Tests

```bash
PYTHONPATH=. pytest -q
```

The current test suite checks memory accounting, recovery latency, P1 feasibility, P2 state constraints, ring direction conventions, and baseline/simulator behavior.
