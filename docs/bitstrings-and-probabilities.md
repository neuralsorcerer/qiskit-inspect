# Bitstrings & Probabilities

## Ordering

- Full distributions for n qubits are rendered as strings "q\_{n-1}…q_0" with MSB on the left (highest qubit index) and LSB on the right (qubit 0).
- Marginals over a list of qubits use that list’s order: the leftmost bit in the key corresponds to the first qubit in the list.
- Passing an empty qubit list represents the trivial marginal and returns `{"": 1.0}` for each prefix.

Tip: To inspect the classical bit order and the qubit→clbit mapping per prefix when using sampler-based tracing, enable logging via `enable_trace_logging()` and pass `debug_bit_order=True` to the trace function.

All probability dictionaries returned by the helpers use plain Python string keys. Numpy scalar strings, integers, tuples of bits, or register-separated labels produced by upstream providers are normalized into canonical bitstrings (left-padded with zeros to the expected number of qubits) so downstream tooling always sees a stable shape.

## Methods

- Exact (no sampling)

  - `trace_probabilities_with_statevector_exact(circuit, include_initial=False, initial_state=None, flatten_control_flow=False)`
  - `trace_marginal_probabilities_with_statevector(circuit, qubits, include_initial=False, initial_state=None, flatten_control_flow=False)`
  - Rounding to 15 decimals removes tiny numerical noise.
  - Marginals are sparse: zero-probability outcomes omitted; on all-zero edge case, argmax is retained.

- SamplerV2

  - `trace_probabilities_with_sampler`: uses `.join_data().get_counts()` (with fallbacks), then normalizes.
  - `trace_counts_with_sampler`: returns raw shot counts with the same canonical key cleanup.
  - `trace_marginal_probabilities_with_sampler`: helper prefixes always end in measurements, remeasuring qubits that were acted upon after their last measurement. Set `add_measure_for_qubits=True` to inject temporary measurements only for the requested qubits; prefixes that still lack any requested qubit return an empty dict.

- Aer simulator
  - `trace_probabilities_with_aer`: sampling approach; increases in shots improve agreement with exact.
  - `trace_counts_with_aer`: thin wrapper over the Aer SamplerV2 that exposes raw counts.
  - See `tests/test_aer_comparison_optional.py` for L1 tolerance scaling with shots.

## Worked examples

### Full distribution ordering

```python
from qiskit import QuantumCircuit
from qiskit_inspect import trace_probabilities_with_statevector_exact

qc = QuantumCircuit(3)
qc.x(2)  # highest index
probs = trace_probabilities_with_statevector_exact(qc)
print(probs[-1])  # {'100': 1.0}  # MSB on left, qubit 2 is leftmost
```

### Marginal ordering and sparsity

```python
from qiskit import QuantumCircuit
from qiskit_inspect import trace_marginal_probabilities_with_statevector

qc = QuantumCircuit(3)
qc.h(0); qc.cx(0,2)
print(trace_marginal_probabilities_with_statevector(qc, [2,0])[-1])
# {'00': 0.5, '11': 0.5}  # first key bit corresponds to qubit 2
```
