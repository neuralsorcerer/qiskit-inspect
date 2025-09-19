"""Trace marginal probabilities and visualise them with ASCII/Matplotlib."""

from qiskit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram

from qiskit_inspect.backend_trace import trace_marginal_probabilities_with_sampler
from qiskit_inspect.visual import ascii_histogram


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def main() -> None:
    qc = build_circuit()
    sampler = StatevectorSampler(default_shots=2048)
    probs_each = trace_marginal_probabilities_with_sampler(
        qc,
        sampler,
        qubits=[0],
        shots=2048,
        add_measure_for_qubits=True,
    )

    for i, p in enumerate(probs_each, 1):
        print(f"Prefix {i}\n{ascii_histogram(p, width=30)}\n")

    # Optional Matplotlib plot for the final prefix using Qiskit's built-in helper
    try:
        plot_histogram(probs_each[-1], title="Marginal on q0 (final prefix)")
    except (MissingOptionalLibraryError, ImportError) as exc:
        print("Matplotlib not installed:", exc)


if __name__ == "__main__":
    main()
