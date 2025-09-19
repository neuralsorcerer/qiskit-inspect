# Getting Started

## Installation

```bash
pip install -e .

# optional extras
pip install -e .[aer]          # Aer-backed sampling helpers
pip install -e .[runtime]      # IBM Runtime SamplerV2 support
pip install -e .[data]         # pandas-powered DataFrame helpers
pip install -e .[visualization]  # Matplotlib/Seaborn utilities
pip install -e .[dev]          # all extras + tooling (pytest, black, ...)
```

Verify that you are using Qiskit 2.0 or newer before running the helpers:

```bash
python - <<'PY'
import qiskit
print(qiskit.__qiskit_version__)
PY
```

> **Note:** Qiskit Inspect requires SamplerV2 primitives and
> ``Statevector.expectation_value`` ``qargs`` support, both of which shipped in
> the Qiskit 2.x line.

## Quickstart

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit import Parameter
from qiskit_inspect import CircuitDebugger
from qiskit_inspect import trace_counts_with_sampler, trace_probabilities_with_sampler

qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])

# Step through the exact state
dbg = CircuitDebugger(qc)
for rec in dbg.trace(include_initial=True):
    print(f"step {rec.step_index}: {rec.instruction} -> bits={rec.classical_bits}")

# Sample per-prefix probabilities
sampler = StatevectorSampler()
probs_per_prefix = trace_probabilities_with_sampler(qc, sampler, shots=1024)
counts_per_prefix = trace_counts_with_sampler(qc, sampler, shots=1024)
print(probs_per_prefix[-1])
print(counts_per_prefix[-1])

# Parameterized circuits: supply values for each prefix when tracing
theta = Parameter("theta")
param_qc = QuantumCircuit(1, 1)
param_qc.ry(theta, 0)
param_qc.measure(0, 0)

param_probs = trace_probabilities_with_sampler(
    param_qc, sampler, shots=1024, parameter_values={theta: 0.5}
)
print(param_probs[-1])

# Sampler-style ``[[...]]`` bindings with a single entry are broadcast to every prefix.
print(
    trace_probabilities_with_sampler(
        param_qc, sampler, shots=1024, parameter_values=[[0.5]]
    )[-1]
)

# Provide scalar values per prefix when working with single-parameter circuits.
print(
    trace_probabilities_with_sampler(
        param_qc, sampler, shots=1024, parameter_values=[0.0, np.pi]
    )[-1]
)
```

Each item yielded by ``CircuitDebugger.trace`` is a ``TraceRecord`` with
``state`` (a ``Statevector`` snapshot), ``classical_bits`` (an ordered tuple of
clbit values), the ``instruction`` that produced the step, and the zero-based
``step_index``. That structure feeds directly into ``trace_records_to_dataframe``
and the CSV/JSON export helpers.
