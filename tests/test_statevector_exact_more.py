from __future__ import annotations

import math
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.controlflow import CASE_DEFAULT, ForLoopOp, SwitchCaseOp, WhileLoopOp
from qiskit.quantum_info import Operator, Pauli, Statevector

from qiskit_inspect import (
    trace_expectations_with_statevector,
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_statevector_exact,
)


def almost_equal_dict(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-12) -> bool:
    if a.keys() != b.keys():
        return False
    return all(math.isclose(a[k], b[k], rel_tol=0, abs_tol=tol) for k in a)


def test_include_initial_and_measure_collapse():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)

    # Exact probabilities; include initial
    probs = trace_probabilities_with_statevector_exact(qc, include_initial=True)

    # Step 0: |00>
    assert probs[0] == {"00": 1.0}
    # Step 1: after H(0): half 00 half 01; statevector keys are in order 'q1q0'
    assert almost_equal_dict(probs[1], {"00": 0.5, "01": 0.5}) is True
    # The exact sparse keys should be 00 and 01 only
    assert set(probs[1].keys()) == {"00", "01"}

    # Step 2: after measure(0->c0), state collapses to |0> with prob 0.5 or |1> with prob 0.5.
    # Our exact trace is post-measure state; q1 unaffected (still 0)
    assert set(probs[2].keys()) <= {"00", "01"}

    # Step 3: CX uses measured q0 value; since state is exact and post-measure, this will flip q1 iff q0==1 leading to keys 00 or 11.
    assert set(probs[3].keys()) <= {"00", "11"}


def test_reset_semantics():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.reset(0)
    probs = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    # initial |0>
    assert probs[0] == {"0": 1.0}
    # after X -> |1>
    assert probs[1] == {"1": 1.0}
    # after reset -> |0>
    assert probs[2] == {"0": 1.0}


def test_marginal_order_and_sparse_filtering():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)

    m20 = trace_marginal_probabilities_with_statevector(qc, [2, 0], include_initial=True)
    # initial
    assert m20[0] == {"00": 1.0}
    # after H: keys '00' and '10' only
    assert set(m20[1].keys()) == {"00", "10"}
    # after CX: keys '00' and '11' only
    assert set(m20[2].keys()) == {"00", "11"}

    # single qubit marginal is sparse with one key when deterministic
    m1 = trace_marginal_probabilities_with_statevector(qc, [1], include_initial=True)
    assert m1[0] == {"0": 1.0}
    assert m1[1] == {"0": 1.0}


def test_expectations_pauli_and_operator_and_qargs():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Global operators on 2 qubits
    specs = [
        ("Z0Z1", Pauli("ZZ")),
        ("X0X1", Pauli("XX")),
    ]
    vals = trace_expectations_with_statevector(qc, specs, include_initial=True)
    # initial |00>: Z0Z1 = 1
    assert math.isclose(vals[0]["Z0Z1"], 1.0, rel_tol=0, abs_tol=1e-12)
    # after H: entanglement building
    # after CX: Bell state |00>+|11>: Z0Z1 = 1, X0X1 = 1
    assert math.isclose(vals[2]["Z0Z1"], 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(vals[2]["X0X1"], 1.0, rel_tol=0, abs_tol=1e-12)

    # Local operator with qargs
    vals_local = trace_expectations_with_statevector(
        qc, [("Z0", Pauli("Z"), [0])], include_initial=True
    )
    assert math.isclose(vals_local[0]["Z0"], 1.0, rel_tol=0, abs_tol=1e-12)


def test_expectations_respect_parameter_binding():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    vals = trace_expectations_with_statevector(
        qc,
        [("Z", Pauli("Z"))],
        parameter_values={theta: math.pi},
    )

    assert len(vals) == len(qc.data)
    assert math.isclose(vals[0]["Z"], -1.0, rel_tol=0, abs_tol=1e-12)


def test_expectations_errors_on_mismatch():
    qc = QuantumCircuit(2)
    # Operator acting on 1 qubit without qargs on a 2-qubit circuit should error
    bad = Operator.from_label("Z")
    try:
        _ = trace_expectations_with_statevector(qc, [("bad", bad)], include_initial=False)
        assert False, "Expected ValueError due to qubit count mismatch"
    except ValueError:
        pass


def test_statevector_trace_handles_for_loop_and_switch_case():
    qc = QuantumCircuit(1, 1)

    body = QuantumCircuit(1)
    body.x(0)
    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, [])

    qc.measure(0, 0)

    case_zero = QuantumCircuit(1, 1)
    case_zero.x(0)
    default_block = QuantumCircuit(1, 1)
    default_block.z(0)
    switch = SwitchCaseOp(qc.cregs[0], [(0, case_zero), (CASE_DEFAULT, default_block)])
    qc.append(switch, qc.qubits, qc.clbits)

    probs = trace_probabilities_with_statevector_exact(qc, include_initial=True)

    # after for loop of two X gates, state returns to |0>, measure collapses to 0, switch applies X
    assert probs[-1] == {"1": 1.0}


def test_statevector_trace_handles_while_loop():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.x(0)
    body.measure(0, 0)

    loop = WhileLoopOp((qc.clbits[0], 1), body)
    qc.append(loop, qc.qubits, qc.clbits)

    probs = trace_probabilities_with_statevector_exact(qc, include_initial=True)

    # After loop the qubit collapses to |0>
    assert probs[-1] == {"0": 1.0}
