<h1 align="center">
Qiskit Inspect
</h1>

<h4 align="center">
Production-ready debugging, tracing, and analysis helpers for <a href="https://www.ibm.com/quantum/qiskit/">Qiskit</a>.
</h4>

<div align="center">

[![Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-blueviolet?logo=qiskit&logoColor=white)](https://www.ibm.com/quantum/ecosystem/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0%2B-purple?logo=qiskit&logoColor=white)](https://www.ibm.com/quantum/qiskit/)
[![Lints](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/lints.yml/badge.svg)](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/lints.yml?query=branch%3Amain)
[![Tests Ubuntu](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-ubuntu.yml?query=branch%3Amain)
[![Tests Windows](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-windows.yml/badge.svg)](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-windows.yml?query=branch%3Amain)
[![Tests MacOS](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-macos.yml/badge.svg)](https://github.com/neuralsorcerer/qiskit-inspect/actions/workflows/test-macos.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

## Overview

`qiskit_inspect` augments Qiskit v2.0+ with practical tooling for stepping through
circuits, tracing prefix probabilities, logging sampler execution, and exporting
structured results. It was built to help practitioners debug classical control
flow, validate sampler output against exact statevectors, and ship robust tests
for production quantum applications.

## Why Qiskit Inspect?

- **Deterministic debugging** – Step through instructions with full
  statevector snapshots, classical bit snapshots, and support for control-flow
  constructs (`IfElseOp`, loops, and `SwitchCaseOp`).
- **Prefix analytics** – Compute exact probabilities, sampler-backed
  probabilities, counts, marginals, expectation values, Shannon entropies,
  divergences (cross entropy, KL, Jensen-Shannon), and total-variation/
  Hellinger distances for every prefix of a circuit. Compare traced prefixes
  against reference distributions with per-step divergence series to spot drift
  quickly.
- **Testing confidence** – Assertion helpers compare counts, probabilities, and
  statevectors with configurable tolerances for noisy hardware.
- **Operational insights** – CSV/JSON export, pandas integration, ASCII/Matplotlib
  histograms, and structured trace logging make it simple to observe and share
  what your primitives are doing.

## Installation

Install via pip

```bash
pip install qiskit-inspect
```

Install the package in editable mode while iterating locally:

```bash
pip install -e .
```

Optional extras install integrations used by some helpers and examples:

- Aer sampling: `pip install qiskit-aer` or `pip install -e .[aer]`
- IBM Runtime SamplerV2: `pip install qiskit-ibm-runtime` or `pip install -e .[runtime]`
- Data wrangling: `pip install pandas` or `pip install -e .[data]`
- Visualization: `pip install matplotlib seaborn` or `pip install -e .[visualization]`
- Development (all extras + tooling): `pip install -e .[dev]`

Install extras with `pip install -e .[extra]` to pull in the optional
dependencies directly from this repository while you iterate locally.

## Quickstart

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_inspect import (
    CircuitDebugger,
    jensen_shannon_divergence,
    prefix_jensen_shannon_divergences,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
    trace_shannon_entropy_with_statevector,
)

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.ry(0.3, 1)
qc.measure(1, 1)

# Step through each prefix deterministically.
debugger = CircuitDebugger(qc)
for record in debugger.trace(include_initial=True):
    print(
        "step",
        record.step_index,
        ":",
        record.instruction,
        "bits=",
        record.classical_bits,
    )

# Compare exact vs sampled prefix probabilities.
exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
sampler = StatevectorSampler(default_shots=2048)
sampled = trace_probabilities_with_sampler(qc, sampler, shots=2048)
print("last-prefix exact probs:", exact[-1])
print("last-prefix sampler probs:", sampled[-1])
entropies = trace_shannon_entropy_with_statevector(qc)
print("prefix entropies:", entropies)
print(
    "final-prefix JS divergence vs uniform:",
    jensen_shannon_divergence(
        exact[-1], {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    ),
)
print(
    "exact vs sampler JS divergence:",
    jensen_shannon_divergence(exact[-1], sampled[-1]),
)
print(
    "per-prefix JS divergence vs sampler:",
    prefix_jensen_shannon_divergences(exact, sampled),
)
```

Set `flatten_control_flow=True` on the exact helpers to emit intermediate
snapshots for operations inside control-flow blocks (for example every
iteration of a `ForLoopOp`).

Every element yielded by `CircuitDebugger.trace()` is a `TraceRecord` exposing
`step_index`, the executed `instruction`, the post-instruction quantum
`state`, and an ordered tuple of `classical_bits`. Those records can be fed
directly into helpers such as `trace_records_to_dataframe` or persisted with
`write_trace_json` for offline analysis.

Continue with the [cookbook](./docs/cookbook.md) for deeper recipes covering
breakpoints, control-flow flattening, expectation values, exporting, pandas
integration, and parameter broadcasting. If you prefer runnable scripts, see the
[examples directory](./examples/).

## Documentation

- [Cookbook](./docs/cookbook.md) – Comprehensive, task-oriented how-to guide.
- [Examples](./docs/examples.md) – Directory tour with command-line entry points.
- [API reference](./docs/api-reference.md) – Detailed API documentation.

## Examples

All scripts live in [`examples/`](./examples/) and can be executed directly:

- `debugger_trace_walkthrough.py` – Structured traces, flattening, and
  breakpoints with trace logging.
- `probabilities_and_counts_walkthrough.py` – Compare exact probabilities,
  sampler counts, marginals, and Aer sampling.
- `expectations_and_exports.py` – Prefix expectation values, DataFrame helpers,
  and CSV/JSON export utilities.
- `parameter_broadcasting.py` – Accepted parameter binding formats for prefix
  tracing helpers.
- `testing_helpers.py` – Assertion utilities for unit tests.
- `bell_backend_trace.py` – Minimal prefix tracing on a Bell circuit.
- `compare_methods.py` – Exact, marginal, and Aer-based probability tracing.
- `breakpoints_demo.py` – Quick look at `run_until_*` helpers.
- `custom_condition_evaluator.py` – Override classical evaluation in the
  debugger.
- `marginal_histograms.py` and `plot_histogram_example.py` – Visualization
  utilities leveraging Qiskit's Matplotlib histogram helper (requires Matplotlib).
- `teleportation_ifelse.py` – Debug an If/Else controlled teleportation circuit.

## Compatibility

- Requires Python 3.10+ and Qiskit 2.0 or newer.
- Works with `SamplerV2`/`EstimatorV2` primitives and classical control-flow
  constructs introduced in Qiskit 2.x.
- Optional extras (Aer, IBM Runtime, pandas, Matplotlib) are auto-detected where
  relevant; helpers degrade gracefully when packages are missing.

## Development

Clone the repository, install with the `dev` extra, and run the test suite:

```bash
pip install -e .[dev]
pytest
```

We welcome issues and pull requests that improve compatibility with the latest
Qiskit releases, add diagnostics that make real-world debugging easier, or
expand the cookbook and examples with practical workflows.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We extend our thanks to the [Qiskit](https://www.ibm.com/quantum/qiskit/) project and [IBM Quantum](https://www.ibm.com/quantum/) team.
