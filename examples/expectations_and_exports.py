from __future__ import annotations

"""Compute prefix expectation values and export results to CSV/JSON."""

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import Pauli

from qiskit_inspect import (
    CircuitDebugger,
    counts_to_dataframe,
    expectations_to_dataframe,
    probabilities_to_dataframe,
    trace_counts_with_sampler,
    trace_expectations_with_estimator,
    trace_expectations_with_statevector,
    trace_probabilities_with_statevector_exact,
    trace_records_to_dataframe,
    write_expectations_csv,
    write_expectations_json,
    write_trace_csv,
    write_trace_json,
)


def build_expectation_circuit() -> QuantumCircuit:
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    return qc


def main() -> None:
    qc = build_expectation_circuit()
    theta = qc.parameters[0]
    theta_value = float(np.pi / 5)
    theta_binding = {theta: theta_value}

    observables = [("<Z>", Pauli("Z"), [0])]
    statevector_rows = trace_expectations_with_statevector(
        qc,
        observables,
        include_initial=True,
        parameter_values=[theta_value],
    )

    estimator = StatevectorEstimator()
    estimator_rows = trace_expectations_with_estimator(
        qc,
        observables,
        estimator=estimator,
        include_initial=True,
        parameter_values=[{theta: 0.0}, {theta: float(np.pi / 2)}],
    )

    output_dir = Path("expectations_output")
    output_dir.mkdir(exist_ok=True)

    write_expectations_csv(statevector_rows, output_dir / "statevector.csv")
    write_expectations_json(statevector_rows, output_dir / "statevector.json")
    write_expectations_csv(estimator_rows, output_dir / "estimator.csv")
    write_expectations_json(estimator_rows, output_dir / "estimator.json")

    qc_for_debug = qc.assign_parameters(theta_binding, inplace=False)
    qc_for_debug.measure_all()
    debugger = CircuitDebugger(qc_for_debug)
    trace_dicts = debugger.trace_as_dicts(include_initial=True, include_markers=True)
    write_trace_csv(trace_dicts, output_dir / "trace.csv")
    write_trace_json(trace_dicts, output_dir / "trace.json")

    trace_df = trace_records_to_dataframe(
        debugger.trace(include_initial=True), classical_bit_columns=["c0"]
    )
    probs_df = probabilities_to_dataframe(
        trace_probabilities_with_statevector_exact(
            qc, include_initial=True, parameter_values=[theta_value]
        )
    )
    qc_for_counts = qc.assign_parameters(theta_binding, inplace=False)
    qc_for_counts.measure_all()
    counts_df = counts_to_dataframe(trace_counts_with_sampler(qc_for_counts, StatevectorSampler()))
    expectations_df = expectations_to_dataframe(statevector_rows)

    print("Trace DataFrame:\n", trace_df.head())
    print("Probabilities DataFrame:\n", probs_df.tail())
    print("Counts DataFrame:\n", counts_df.tail())
    print("Expectations DataFrame:\n", expectations_df.tail())


if __name__ == "__main__":
    main()
