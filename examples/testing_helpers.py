from __future__ import annotations

"""Sanity-check circuits using the qiskit_inspect assertion helpers."""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from qiskit_inspect import (
    assert_counts_close,
    assert_probabilities_close,
    assert_state_equiv,
    trace_counts_with_sampler,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)


def main() -> None:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=2048)
    sampled_probs = trace_probabilities_with_sampler(qc, sampler, shots=2048)
    sampled_counts = trace_counts_with_sampler(qc, sampler, shots=2048)
    exact_probs = trace_probabilities_with_statevector_exact(qc)

    assert_probabilities_close(sampled_probs[-1], exact_probs[-1], tol_l1=0.1)
    assert_counts_close(sampled_counts[-1], exact_probs[-1], shots=2048, tol_l1=0.1)

    bell = QuantumCircuit(1)
    bell.h(0)
    sv0 = Statevector.from_label("0").evolve(bell)
    sv1 = Statevector.from_label("+")
    assert_state_equiv(sv0, sv1)

    print("Assertions completed successfully.")


if __name__ == "__main__":
    main()
