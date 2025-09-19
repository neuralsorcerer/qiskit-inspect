"""Trace a Bell circuit with SamplerV2 and (optionally) Aer."""

from qiskit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.primitives import StatevectorSampler

from qiskit_inspect import trace_probabilities_with_aer, trace_probabilities_with_sampler


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def main() -> None:
    qc = build_circuit()

    # Preferred SamplerV2 path (portable across local sim, Aer, Runtime)
    sampler = StatevectorSampler(default_shots=4096)
    probs_each = trace_probabilities_with_sampler(qc, sampler)
    for i, p in enumerate(probs_each, 1):
        print(f"[SamplerV2] Step {i}: {dict(sorted(p.items()))}")

    # Aer fallback (requires qiskit-aer)
    try:
        probs_each_aer = trace_probabilities_with_aer(qc, shots=8192)
    except (MissingOptionalLibraryError, ImportError) as exc:
        print("Aer not available:", exc)
    else:
        for i, p in enumerate(probs_each_aer, 1):
            print(f"[Aer] Step {i}: {dict(sorted(p.items()))}")


if __name__ == "__main__":
    main()
