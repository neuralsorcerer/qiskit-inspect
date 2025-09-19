from __future__ import annotations

"""Show how tracing helpers accept multiple parameter binding formats."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler

from qiskit_inspect import (
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)


def main() -> None:
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=1024)

    formats = {
        "dict": {theta: np.pi / 8},
        "list-of-dicts": [{theta: np.pi / 4}],
        "list": [0.0, np.pi],
        "sampler-style": [[np.pi / 3]],
    }

    for label, params in formats.items():
        sampled = trace_probabilities_with_sampler(
            qc,
            sampler,
            shots=1024,
            parameter_values=params,
        )
        print(f"{label} bindings (last prefix): {sampled[-1]}")

    exact = trace_probabilities_with_statevector_exact(
        qc,
        include_initial=True,
        parameter_values=[np.pi / 8],
    )
    print("Statevector exact probabilities (last prefix):", exact[-1])


if __name__ == "__main__":
    main()
