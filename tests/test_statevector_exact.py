from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.controlflow import ForLoopOp
from qiskit.quantum_info import Statevector

from qiskit_inspect import (
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_statevector_exact,
    trace_statevectors_with_statevector_exact,
)


def probs_to_tuple(d):
    # deterministic ordering for asserts
    return tuple(sorted(d.items()))


def state_equiv(a: Statevector, b: Statevector) -> bool:
    """Return ``True`` when two statevectors are equal up to global phase."""

    return a.equiv(b)


def _for_loop_two_x_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    body = QuantumCircuit(1)
    body.x(0)
    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, [])
    return qc


def test_trace_statevectors_exact_flatten_control_flow():
    qc = _for_loop_two_x_circuit()

    states = trace_statevectors_with_statevector_exact(
        qc,
        include_initial=True,
        flatten_control_flow=True,
    )

    assert len(states) == 4  # initial + two X gates + for_loop record
    expected = ["0", "1", "0", "0"]
    for got, label in zip(states, expected):
        assert state_equiv(got, Statevector.from_label(label))


def test_trace_statevectors_exact_includes_initial_and_prefix_states():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.z(0)

    states = trace_statevectors_with_statevector_exact(qc, include_initial=True)

    assert len(states) == len(qc.data) + 1
    assert state_equiv(states[0], Statevector.from_label("0"))
    assert state_equiv(states[1], Statevector.from_label("+"))
    assert state_equiv(states[2], Statevector.from_label("-"))


def test_trace_statevectors_exact_respects_parameter_binding():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    states = trace_statevectors_with_statevector_exact(
        qc,
        include_initial=False,
        parameter_values={theta: math.pi},
    )

    assert len(states) == len(qc.data)
    assert state_equiv(states[0], Statevector.from_label("1"))


def test_trace_probabilities_exact_rejects_flatten_control_flow():
    qc = _for_loop_two_x_circuit()

    with pytest.raises(ValueError, match="flatten_control_flow"):
        trace_probabilities_with_statevector_exact(
            qc,
            include_initial=True,
            flatten_control_flow=True,
        )


def test_exact_probabilities_simple_h():
    qc = QuantumCircuit(1)
    qc.h(0)
    res = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    assert all(isinstance(k, str) for k in res[0])
    assert all(isinstance(k, str) for k in res[1])
    # initial -> |0>
    assert math.isclose(res[0].get("0", 0.0), 1.0, rel_tol=0, abs_tol=1e-12)
    # after H -> equal superposition
    assert math.isclose(res[1]["0"], 0.5, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(res[1]["1"], 0.5, rel_tol=0, abs_tol=1e-12)


def test_exact_marginals_flatten_control_flow():
    qc = _for_loop_two_x_circuit()

    marg = trace_marginal_probabilities_with_statevector(
        qc,
        [0],
        include_initial=True,
        flatten_control_flow=True,
    )

    assert marg == [{"0": 1.0}, {"1": 1.0}, {"0": 1.0}, {"0": 1.0}]


def test_exact_probabilities_respect_initial_state():
    qc = QuantumCircuit(1)
    qc.x(0)
    res = trace_probabilities_with_statevector_exact(
        qc,
        include_initial=True,
        initial_state=Statevector.from_label("1"),
    )
    assert probs_to_tuple(res[0]) == (("1", 1.0),)
    assert probs_to_tuple(res[1]) == (("0", 1.0),)


def test_exact_probabilities_parameter_binding_mapping():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    res = trace_probabilities_with_statevector_exact(
        qc,
        parameter_values={theta: 0.0},
    )

    assert len(res) == len(qc.data)
    assert probs_to_tuple(res[0]) == (("0", 1.0),)


def test_exact_probabilities_parameter_binding_sequence():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    res = trace_probabilities_with_statevector_exact(
        qc,
        parameter_values=[0.0],
    )

    assert len(res) == len(qc.data)
    assert probs_to_tuple(res[0]) == (("0", 1.0),)


def test_exact_probabilities_parameter_binding_nested_sequence():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    res = trace_probabilities_with_statevector_exact(
        qc,
        parameter_values=[[0.0]],
    )

    assert len(res) == len(qc.data)
    assert probs_to_tuple(res[0]) == (("0", 1.0),)


def test_exact_probabilities_skip_measure_when_condition_false():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    meas_ci = qc.data[-1]
    mutable = meas_ci.operation.to_mutable()
    mutable.condition = (qc.cregs[0], 1)
    qc.data[-1] = meas_ci.replace(operation=mutable)

    res = trace_probabilities_with_statevector_exact(qc, include_initial=True)

    assert len(res) == len(qc.data) + 1
    # initial state |0>
    assert probs_to_tuple(res[0]) == (("0", 1.0),)
    # after H -> superposition
    assert probs_to_tuple(res[1]) == (("0", 0.5), ("1", 0.5))
    # measurement condition false, probabilities unchanged
    assert set(res[2]) == {"0", "1"}
    assert math.isclose(res[2]["0"], 0.5, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(res[2]["1"], 0.5, rel_tol=0, abs_tol=1e-12)


def test_exact_probabilities_reject_mismatched_initial_state_qubits():
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="acts on 2 qubits"):
        trace_probabilities_with_statevector_exact(qc, initial_state=Statevector.from_label("00"))


def test_exact_probabilities_reject_initial_state_non_power_of_two():
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="not a power of two"):
        trace_probabilities_with_statevector_exact(qc, initial_state=[1, 0, 0])


def test_exact_marginals_two_qubits():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # Full probs after each step
    full = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    # Marginal on qubit[1] only should match distribution of that bit
    marg = trace_marginal_probabilities_with_statevector(qc, [1], include_initial=True)

    # initial
    assert probs_to_tuple(marg[0]) == (("0", 1.0),)
    # after H(0): q1 is still 0
    assert probs_to_tuple(marg[1]) == (("0", 1.0),)
    # after CX: q1 equals q0, so half 0, half 1
    assert probs_to_tuple(marg[2]) == (("0", 0.5), ("1", 0.5))


def test_exact_marginal_order_of_qubits():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    # Marginal over [2,0] should format keys with qubit 2 as MSB then qubit 0
    marg = trace_marginal_probabilities_with_statevector(qc, [2, 0], include_initial=False)
    # After first step H(0), qubit 2 is 0, qubit 0 is equally 0/1 -> keys '00' and '10'
    assert set(marg[0].keys()) == {"00", "10"}
    # After CX(0->2): perfectly correlated -> '00' and '11'
    assert set(marg[1].keys()) == {"00", "11"}


def test_exact_marginals_respect_qubit_order():
    qc = QuantumCircuit(2)

    results = trace_marginal_probabilities_with_statevector(
        qc,
        [0, 1],
        include_initial=True,
        initial_state=Statevector.from_label("01"),
    )

    assert results == [{"01": 1.0}]

    swapped = trace_marginal_probabilities_with_statevector(
        qc,
        [1, 0],
        include_initial=True,
        initial_state=Statevector.from_label("01"),
    )

    assert swapped == [{"10": 1.0}]


def test_exact_marginals_respect_initial_state():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    marg = trace_marginal_probabilities_with_statevector(
        qc,
        [1],
        include_initial=True,
        initial_state=Statevector.from_label("10"),
    )
    assert probs_to_tuple(marg[0]) == (("1", 1.0),)
    assert probs_to_tuple(marg[1]) == (("1", 1.0),)


def test_exact_marginals_respect_parameter_binding():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    marg = trace_marginal_probabilities_with_statevector(
        qc,
        [0],
        parameter_values={theta: math.pi},
    )

    assert len(marg) == len(qc.data)
    entry = marg[0]
    assert math.isclose(entry.get("1", 0.0), 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(entry.get("0", 0.0), 0.0, rel_tol=0, abs_tol=1e-12)


def test_exact_marginals_validate_qubit_indices():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="Invalid qubit index"):
        trace_marginal_probabilities_with_statevector(qc, [2])
    with pytest.raises(ValueError, match="Duplicate qubit index"):
        trace_marginal_probabilities_with_statevector(qc, [0, 0])


def test_exact_marginals_allow_empty_qubit_list():
    qc = QuantumCircuit(1)
    qc.h(0)
    res = trace_marginal_probabilities_with_statevector(qc, [], include_initial=True)
    # initial and after the H gate both represent the trivial marginal
    assert res == [{"": 1.0}, {"": 1.0}]
