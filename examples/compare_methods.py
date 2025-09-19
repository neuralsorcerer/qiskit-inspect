from __future__ import annotations

"""Compare exact statevector tracing vs Aer SamplerV2 sampling.

This script prints per-prefix exact probabilities, exact marginals, and (when
available) qiskit-aer ``SamplerV2`` samples for the same circuit.  The helper
functions work on Qiskit 2.0+ and require the optional ``qiskit-aer`` extra to
demonstrate backend sampling.
"""

from qiskit import QuantumCircuit

from qiskit_inspect import (
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_aer,
    trace_probabilities_with_statevector_exact,
)


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.cx(1, 2)
    return qc


def main():
    qc = build_circuit()

    print("Circuit:\n", qc)

    # Exact probabilities per prefix
    exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    print("\nExact probabilities (include_initial=True):")
    for i, p in enumerate(exact):
        print(f"step {i}: {p}")

    # Exact marginals for selected qubits (e.g., [2, 0])
    marg = trace_marginal_probabilities_with_statevector(qc, [2, 0], include_initial=True)
    print("\nExact marginal probs for qubits [2,0] (include_initial=True):")
    for i, p in enumerate(marg):
        print(f"step {i}: {p}")

    # Aer-based sampling (if Aer is available).
    # Note: requires installing the optional dependency qiskit-aer.
    try:
        aer_probs = trace_probabilities_with_aer(
            qc, shots=8192, method="automatic", debug_bit_order=False
        )
        print("\nAer-sampled probabilities (shots=8192):")
        for i, p in enumerate(aer_probs):
            print(f"prefix {i+1}: {p}")
    except Exception as exc:
        print("\nAer sampling not available:", exc)


if __name__ == "__main__":
    main()
