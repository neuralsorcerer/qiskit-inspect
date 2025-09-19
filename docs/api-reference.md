# API Reference

## Debugger

- `CircuitDebugger`
  - `.trace(include_initial=True, include_markers=False)`
  - `.trace_as_dicts(state_format='probs' | 'amplitudes', include_markers=False)`
  - `.run_until(predicate)` / `.run_until_op(name)` / `.run_until_index(i)`

## Backend tracing

- `trace_probabilities_with_sampler(circuit, sampler, shots=4096, debug_bit_order=False, parameter_values=None)`
- `trace_counts_with_sampler(circuit, sampler, shots=4096, debug_bit_order=False, parameter_values=None)`
- `trace_probabilities_with_aer(circuit, shots=8192, method='automatic', debug_bit_order=False, parameter_values=None)`
- `trace_counts_with_aer(circuit, shots=8192, method='automatic', debug_bit_order=False, parameter_values=None)`
- `trace_probabilities_with_statevector_exact(circuit, include_initial=False, initial_state=None, parameter_values=None, flatten_control_flow=False)`
- `trace_statevectors_with_statevector_exact(circuit, include_initial=False, initial_state=None, parameter_values=None, flatten_control_flow=False)`
- `trace_marginal_probabilities_with_sampler(circuit, sampler, qubits, shots=4096, add_measure_for_qubits=False, debug_bit_order=False, parameter_values=None)`
- `trace_marginal_probabilities_with_statevector(circuit, qubits, include_initial=False, initial_state=None, parameter_values=None, flatten_control_flow=False)`
- `trace_expectations_with_statevector(circuit, observables, initial_state=None, include_initial=False, parameter_values=None)`
-   Observables accept ``Operator``, ``SparsePauliOp``, ``Pauli`` or Pauli-string inputs.
- `trace_expectations_with_estimator(circuit, observables, estimator, include_initial=False, parameter_values=None, precision=None)`
  - Requires measurement-free circuits; measurement instructions raise ``ValueError``.

## Analytics

- `shannon_entropy(probabilities, base=2.0, num_qubits=None)` – Shannon entropy of a
  probability/counts mapping.
- `prefix_shannon_entropies(prefix_probabilities, base=2.0, num_qubits=None)` – Entropy
  per prefix probability dictionary.
- `total_variation_distance(first, second, num_qubits=None)` – Total-variation distance
  between two distributions.
- `cross_entropy(first, second, base=2.0, num_qubits=None)` – Cross entropy of two
  distributions.
- `kullback_leibler_divergence(first, second, base=2.0, num_qubits=None)` – KL
  divergence ``D_KL(first || second)``.
- `jensen_shannon_divergence(first, second, base=2.0, num_qubits=None)` –
  Jensen-Shannon divergence between two distributions.
- `hellinger_distance(first, second, num_qubits=None)` – Hellinger distance between two
  distributions.
- `prefix_total_variation_distances(prefix_probabilities, reference, num_qubits=None)` –
  Total-variation distance per prefix against a reference distribution or sequence.
- `prefix_cross_entropies(prefix_probabilities, reference, base=2.0, num_qubits=None)` –
  Cross entropy per prefix against reference distributions.
- `prefix_kullback_leibler_divergences(prefix_probabilities, reference, base=2.0, num_qubits=None)` –
  KL divergence per prefix against reference distributions.
- `prefix_jensen_shannon_divergences(prefix_probabilities, reference, base=2.0, num_qubits=None)` –
  Jensen-Shannon divergence per prefix against reference distributions.
- `prefix_hellinger_distances(prefix_probabilities, reference, num_qubits=None)` –
  Hellinger distance per prefix against reference distributions.
- `trace_shannon_entropy_with_statevector(circuit, include_initial=False, initial_state=None, parameter_values=None, base=2.0)`
  – Prefix entropies using the exact statevector debugger.
- `trace_shannon_entropy_with_sampler(circuit, sampler, shots=4096, debug_bit_order=False, parameter_values=None, base=2.0)`
  – Prefix entropies using a SamplerV2 backend.

## Visuals & Export

- Visuals: `pretty_ket`, `top_amplitudes`, `ascii_histogram`
- For Matplotlib histogram plots, use `qiskit.visualization.plot_histogram` from Qiskit itself.
- Export:
  - `write_expectations_csv/json`, `write_trace_csv/json`
  - DataFrame helpers: `trace_records_to_dataframe(records, state_format='probs'|'amplitudes', classical_bits=True, classical_bit_columns=False)`,
    `probabilities_to_dataframe`, `counts_to_dataframe`, `expectations_to_dataframe`

## Assertions

- `assert_state_equiv`
- `assert_probabilities_close`
- `assert_counts_close`
