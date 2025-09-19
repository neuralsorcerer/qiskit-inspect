from __future__ import annotations

"""Trace probabilities, counts, and marginals with samplers and exact methods."""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from qiskit_inspect import (
    ascii_histogram,
    trace_counts_with_sampler,
    trace_marginal_probabilities_with_sampler,
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_aer,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="probability_demo")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.ry(0.3, 1)
    qc.measure(1, 1)
    return qc


def main() -> None:
    qc = build_circuit()
    sampler = StatevectorSampler(default_shots=2048)

    exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    sampled = trace_probabilities_with_sampler(qc, sampler, shots=2048)
    counts = trace_counts_with_sampler(qc, sampler, shots=2048)

    print("Exact last-prefix probabilities:", dict(sorted(exact[-1].items())))
    print("Sampler last-prefix probabilities:", dict(sorted(sampled[-1].items())))
    print("Sampler last-prefix counts:", dict(sorted(counts[-1].items())))
    print("ASCII histogram:\n", ascii_histogram(exact[-1]))

    marg_exact = trace_marginal_probabilities_with_statevector(qc, qubits=[1], include_initial=True)
    marg_sampled = trace_marginal_probabilities_with_sampler(
        qc,
        sampler,
        qubits=[1],
        shots=2048,
        add_measure_for_qubits=True,
    )
    print("Exact marginals for qubit 1:", marg_exact[-1])
    print("Sampled marginals for qubit 1:", marg_sampled[-1])

    try:
        aer_probs = trace_probabilities_with_aer(qc, shots=4096)
        print("Aer last-prefix probabilities:", dict(sorted(aer_probs[-1].items())))
    except Exception as exc:
        print("Aer helpers unavailable:", exc)


if __name__ == "__main__":
    main()
