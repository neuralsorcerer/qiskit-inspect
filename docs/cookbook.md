# Cookbook

Task-oriented recipes demonstrating the full surface area of `qiskit_inspect`.
Each section links core helpers with common workflows on Qiskit 2.0+ circuits.

## Debugging circuits

### Step through instructions with `CircuitDebugger`

```python
from qiskit import QuantumCircuit
from qiskit_inspect import CircuitDebugger, format_classical_bits, pretty_ket

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
with qc.if_test((qc.clbits[0], 1)):
    qc.z(1)
qc.measure(1, 1)

debugger = CircuitDebugger(qc)
for record in debugger.trace(include_initial=True):
    print(
        f"step {record.step_index}: {record.instruction!s:<15}"
        f" bits={format_classical_bits(record.classical_bits)}"
        f" state={pretty_ket(record.state)}"
    )
```

- `include_initial=True` yields the all-zero prefix at index 0.
- `TraceRecord.state` exposes a `Statevector`, while `TraceRecord.classical_bits`
  is an ordered tuple matching the circuit's classical registers.
- `if_test` requires a `(clbit, value)` tuple in Qiskit 2.0+, so `qc.clbits[0]`
  selects the first classical bit for the branch condition.

### Flatten nested control flow and include markers

```python
flat = debugger.trace(
    include_initial=True,
    include_markers=True,
    flatten_control_flow=True,
)
for record in flat:
    marker = f"[{record.marker}]" if record.marker else "-"
    print(f"step {record.step_index}: {record.instruction!s:<15} marker={marker}")
```

Markers show loop boundaries and `IfElseOp` entry/exit points. Flattening turns
nested instructions into a linear stream that mirrors the execution order of
unrolled programs.

### Breakpoints with `run_until_*`

```python
snap_op = debugger.run_until_op("cx")      # stop when next CX executes
snap_idx = debugger.run_until_index(4)      # stop when step_index == 4
print("first CX instruction:", snap_op[-1].instruction)
print("state after step 4:", pretty_ket(snap_idx[-1].state))
```

Call `debugger.reset()` before chaining multiple breakpoint queries to restart
from the beginning of the circuit.

### Custom condition evaluation

Override classical expressions used in `IfElseOp`/loop conditions by supplying
`condition_evaluator` to the constructor. See
[`examples/custom_condition_evaluator.py`](../examples/custom_condition_evaluator.py)
for a full demonstration.

## Probability, count, and marginal tracing

Use the same `qc` defined in the debugging section (or substitute your own circuit).

### Exact and sampled prefix probabilities

```python
from qiskit.primitives import StatevectorSampler
from qiskit_inspect import (
    ascii_histogram,
    trace_counts_with_sampler,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)

sampler = StatevectorSampler(default_shots=4096)
exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
sampled = trace_probabilities_with_sampler(qc, sampler, shots=4096)
counts = trace_counts_with_sampler(qc, sampler, shots=4096)

print("last-prefix exact:", dict(sorted(exact[-1].items())))
print("last-prefix sampler:", dict(sorted(sampled[-1].items())))
print("last-prefix counts:", dict(sorted(counts[-1].items())))
print(ascii_histogram(exact[-1]))
```

- Exact tracing simulates the statevector without sampling noise.
- Sampler-based helpers accept any `SamplerV2` backend (including IBM Runtime).
- When Aer is installed, `trace_probabilities_with_aer` and
  `trace_counts_with_aer` provide hardware-accurate sampling. Exceptions are
  surfaced gracefully if Aer is unavailable.

### Start from custom initial states

```python
from qiskit.quantum_info import Statevector

exact = trace_probabilities_with_statevector_exact(
    qc,
    initial_state=Statevector.from_label("11"),
)
print(exact[0])  # {'11': 1.0}
```

Initial states must match the circuit's qubit count and have power-of-two
dimension. Bitstring inputs are stripped of whitespace/underscores but must
supply one digit per qubit. Invalid inputs raise `ValueError`.

### Marginals over selected qubits

```python
from qiskit_inspect import (
    trace_marginal_probabilities_with_sampler,
    trace_marginal_probabilities_with_statevector,
)

# Exact marginals
marg_exact = trace_marginal_probabilities_with_statevector(qc, qubits=[1, 0])
print(marg_exact[-1])

# Sampler-backed marginals. Use add_measure_for_qubits=True to auto-measure.
marg_sampled = trace_marginal_probabilities_with_sampler(
    qc,
    sampler,
    qubits=[1],
    shots=4096,
    add_measure_for_qubits=True,
)
print(marg_sampled[-1])
```

### Trace logging for sampler helpers

```python
import logging
from qiskit_inspect import enable_trace_logging, trace_probabilities_with_sampler

logger = enable_trace_logging(level=logging.DEBUG)
logger.debug("prefix tracing started")
trace_probabilities_with_sampler(qc, sampler, debug_bit_order=True)
```

Trace logs describe classical bit ordering, prefix metadata, and optional
backend diagnostics. Logs are emitted to stderr; disable by removing the helper.

## Prefix analytics

Quantify the randomness or drift of prefix distributions using the analytics
helpers. Entropies accept raw counts or probability dictionaries, making them a
lightweight diagnostic when tuning control flow.

```python
from qiskit_inspect import (
    hellinger_distance,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    prefix_hellinger_distances,
    prefix_jensen_shannon_divergences,
    prefix_kullback_leibler_divergences,
    prefix_shannon_entropies,
    prefix_total_variation_distances,
    total_variation_distance,
    trace_shannon_entropy_with_sampler,
    trace_shannon_entropy_with_statevector,
)

# Exact prefix entropies using the deterministic debugger
exact_entropies = trace_shannon_entropy_with_statevector(qc, include_initial=True)
print("exact entropies:", exact_entropies)

# Sampler-backed entropies reuse the same backend jobs as probability tracing
sampled_entropies = trace_shannon_entropy_with_sampler(qc, sampler, shots=4096)
print("sampled entropies:", sampled_entropies)

# Aggregate metrics on the final prefix distribution
tv_distance = total_variation_distance(exact[-1], sampled[-1])
js_div = jensen_shannon_divergence(exact[-1], sampled[-1])
kl_div = kullback_leibler_divergence(exact[-1], sampled[-1])
hellinger = hellinger_distance(exact[-1], sampled[-1])
print(f"total-variation distance between exact and sampled: {tv_distance:.3f}")
print(f"Jensen-Shannon divergence: {js_div:.4f} bits")
print(f"KL divergence (exact || sampled): {kl_div:.4f} bits")
print(f"Hellinger distance: {hellinger:.4f}")

# Series metrics show how drift evolves across prefixes.
tv_series = prefix_total_variation_distances(exact, sampled)
js_series = prefix_jensen_shannon_divergences(exact, sampled)
kl_series = prefix_kullback_leibler_divergences(exact, sampled)
hellinger_series = prefix_hellinger_distances(exact, sampled)
print("per-prefix TV distance vs sampler:", tv_series)
print("per-prefix JS divergence vs sampler:", js_series)
print("per-prefix KL divergence vs sampler:", kl_series)
print("per-prefix Hellinger distance vs sampler:", hellinger_series)
```

`prefix_shannon_entropies` accepts any sequence of mappings. For example,
pass a sliding window of `counts` dictionaries to observe how each step's
randomness evolves across calibrations. Reuse the `qc`, `sampler`, `exact`, and
`sampled` variables from the probability section above. Divergence helpers share
the same normalization rules so you can compare exact vs sampled prefixes (or
calibration runs) without pre-processing bitstrings. When you do not have a
reference for a particular prefix, pass `None` in the reference sequence to
receive `math.nan` for that entry while still evaluating the rest.

## Expectation values and observables

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli
from qiskit_inspect import (
    trace_expectations_with_estimator,
    trace_expectations_with_statevector,
)

theta = Parameter("theta")
exp_circuit = QuantumCircuit(1)
exp_circuit.ry(theta, 0)

observables = [("<Z>", Pauli("Z"), [0])]
rows_sv = trace_expectations_with_statevector(
    exp_circuit,
    observables,
    include_initial=True,
    parameter_values=[np.pi / 3],
)
rows_est = trace_expectations_with_estimator(
    exp_circuit,
    observables,
    estimator=StatevectorEstimator(),
    include_initial=True,
    parameter_values=[{theta: 0.0}, {theta: np.pi / 2}],
)
```

- Observables accept `Operator`, `Pauli`, or `SparsePauliOp` values and require
  Hermitian matrices. Non-Hermitian inputs raise `ValueError`.
- Provide explicit `qargs` when your observable acts on a subset of qubits. Qubit
  indices must be unique and within range.
- Estimator circuits must be measurement-free; fall back to the statevector
  helper otherwise.

## Data export and pandas integration

Reuse `qc`, `sampler`, and expectation rows from the preceding sections (or substitute your own data).

```python
from pathlib import Path
from qiskit.primitives import StatevectorSampler
from qiskit_inspect import (
    CircuitDebugger,
    counts_to_dataframe,
    expectations_to_dataframe,
    probabilities_to_dataframe,
    trace_counts_with_sampler,
    trace_probabilities_with_statevector_exact,
    trace_records_to_dataframe,
    write_expectations_csv,
    write_expectations_json,
    write_trace_csv,
    write_trace_json,
)

output_dir = Path("inspect_exports")
output_dir.mkdir(exist_ok=True)

# Trace snapshots as dictionaries then persist to disk.
debug_qc = qc.copy()
debug_qc.measure_all()
debugger = CircuitDebugger(debug_qc)
trace_rows = debugger.trace_as_dicts(
    include_initial=True,
    include_markers=True,
    include_pre_measurement=True,
)
write_trace_csv(trace_rows, output_dir / "trace.csv")
write_trace_json(trace_rows, output_dir / "trace.json")

# Expectation rows from previous section can be exported the same way.
write_expectations_csv(rows_sv, output_dir / "expectations.csv")
write_expectations_json(rows_sv, output_dir / "expectations.json")

# Load results into pandas DataFrames (requires the optional pandas extra).
trace_df = trace_records_to_dataframe(
    debugger.trace(include_initial=True),
    classical_bit_columns=["c0"],
    include_pre_measurement=True,
)
prob_df = probabilities_to_dataframe(
    trace_probabilities_with_statevector_exact(qc, include_initial=True)
)
counts_df = counts_to_dataframe(
    trace_counts_with_sampler(debug_qc, StatevectorSampler())
)
expect_df = expectations_to_dataframe(rows_sv)
```

- Pass `classical_bit_columns=True` (or a list of names) to split the classical
  snapshot tuple into individual nullable integer columns.
- Use `include_pre_measurement=True` to preserve the pre-collapse state for
  measurement records in exported traces and DataFrames. Rows without a
  pre-measurement snapshot leave the CSV cells blank and populate `pd.NA`
  entries in DataFrame exports so callers can distinguish missing data from
  genuine zero probabilities.
- Export helpers accept iterables of dictionaries or `TraceRecord` objects.
- Files are written in UTF-8 with newline termination suitable for git diffs.

## Assertion helpers for tests

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_inspect import (
    assert_counts_close,
    assert_probabilities_close,
    assert_state_equiv,
)

bell = QuantumCircuit(1)
bell.h(0)
sv0 = Statevector.from_label("0").evolve(bell)
sv1 = Statevector.from_label("+")
assert_state_equiv(sv0, sv1)

counts = trace_counts_with_sampler(qc, StatevectorSampler(default_shots=2048))[-1]
probs = trace_probabilities_with_statevector_exact(qc)[-1]
assert_probabilities_close(probs, probs, tol_l1=1e-9)
assert_counts_close(counts, probs, shots=2048, tol_l1=0.1)
```

`tol_l1` bounds the total variation distance; tune it to account for sampling
noise. Assertion helpers raise `AssertionError` with human-readable diffs when
thresholds are exceeded.

## Parameter broadcasting and bindings

The tracing helpers accept the same parameter binding formats as Qiskit's
primitives. A single binding is automatically applied to every prefix unless you
explicitly pass one binding per prefix.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
from qiskit_inspect import trace_probabilities_with_sampler

sampler = StatevectorSampler(default_shots=1024)
theta = Parameter("theta")
param_circuit = QuantumCircuit(1, 1)
param_circuit.ry(theta, 0)
param_circuit.measure(0, 0)

bindings = [
    {theta: np.pi / 8},      # dict
    [{theta: np.pi / 4}],    # list-of-dicts
    [0.0, np.pi],            # flat list, single parameter -> per prefix
    [[np.pi / 3]],           # sampler-style [[...]]
]

for values in bindings:
    rows = trace_probabilities_with_sampler(
        param_circuit,
        sampler,
        shots=1024,
        parameter_values=values,
    )
    print(rows[-1])
```

- Flat lists are interpreted in the circuit's parameter order. For multi-
  parameter circuits, provide one scalar per parameter; the same list is reused
  for prefixes that touch only a subset of parameters.
- Dictionary bindings are filtered per prefix, so you can provide the full
  circuit mapping once.
- When you need different values per prefix, pass a sequence (list/tuple) with
  one entry per prefix. The helpers validate lengths and raise `ValueError` when
  counts mismatch.
