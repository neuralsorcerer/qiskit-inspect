import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector

from qiskit_inspect import (
    CircuitDebugger,
    format_classical_bits,
    trace_expectations_with_estimator,
    trace_expectations_with_statevector,
)


class _FakeJob:
    def __init__(self, payload):
        self._payload = payload

    def result(self):
        return self._payload


class _FakeEstimator:
    def __init__(self, evs):
        self._evs = evs

    def run(self, pubs, **kwargs):
        fake_data = type("FakeData", (), {"evs": self._evs})()
        fake_result = type("FakeResult", (), {"data": fake_data})()
        return _FakeJob([fake_result])


def test_format_classical_bits():
    bits = [None, 1, 0, None]
    assert format_classical_bits(bits) == "x10x"
    assert format_classical_bits(bits, unknown="?") == "?10?"


def test_get_register_value_and_json_export():
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(1)
    qc.add_register(cr)
    qc.x(0)
    qc.measure(0, cr[0])
    dbg = CircuitDebugger(qc)
    recs = dbg.trace()
    # After first step (x), classical bits unknown
    assert recs[0].classical_bits == [None, None]
    # After measure, first bit is 1
    assert recs[-1].classical_bits[0] in (0, 1)
    # JSON export shape
    dicts = dbg.trace_as_dicts()
    assert isinstance(dicts[0]["state"], dict)
    assert all(isinstance(k, str) for k in dicts[0]["state"].keys())


def test_expectation_trace_pauli_z():
    qc = QuantumCircuit(1)
    qc.h(0)
    # After H, <Z> = 0 on |+>
    vals = trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), [0])])
    assert len(vals) == 1
    assert abs(vals[0]["Z"]) < 1e-9


def test_expectation_trace_two_qubits():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # On Bell state, ZI has 0 expectation, ZZ has 1
    vals = trace_expectations_with_statevector(
        qc,
        [
            ("ZI", Pauli("ZI"), [0, 1]),
            ("ZZ", Pauli("ZZ"), [0, 1]),
        ],
    )
    assert len(vals) == 2
    # After H: ZZ undefined on 2 qubits unless both applied; still compute after CX
    assert abs(vals[-1]["ZI"]) < 1e-9
    assert abs(vals[-1]["ZZ"] - 1.0) < 1e-9


def test_expectation_trace_validates_qargs():
    qc = QuantumCircuit(2)
    qc.h(0)

    with pytest.raises(ValueError, match="invalid index 2"):
        trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), [2])])

    with pytest.raises(ValueError, match="duplicate index"):
        trace_expectations_with_statevector(qc, [("ZZ", Pauli("ZZ"), [0, 0])])

    with pytest.raises(ValueError, match="must be integers"):
        trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), ["0"])])

    with pytest.raises(ValueError, match="must be integers"):
        trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), [True])])


def test_expectation_trace_accepts_integral_qargs():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    np_indices = [np.int64(0), np.int32(1)]
    rows = trace_expectations_with_statevector(qc, [("ZZ", Pauli("ZZ"), np_indices)])
    assert pytest.approx(rows[-1]["ZZ"]) == 1.0


def test_expectation_trace_accepts_sparse_pauliop_statevector():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    observables = [
        ("ZZ", SparsePauliOp.from_list([("ZZ", 1.0)])),
        ("Z0", SparsePauliOp.from_list([("Z", 1.0)]), [0]),
    ]

    rows = trace_expectations_with_statevector(qc, observables, include_initial=True)

    assert pytest.approx(rows[-1]["ZZ"], abs=1e-12) == 1.0
    # Expectation of Z on qubit 0 for Bell state is zero.
    assert pytest.approx(rows[-1]["Z0"], abs=1e-12) == 0.0


def test_expectation_trace_requires_matching_operator_width():
    qc = QuantumCircuit(2)
    qc.h(0)

    single_qubit_op = Operator(np.eye(2))
    with pytest.raises(ValueError, match="acts on 1 qubits; please provide qargs"):
        trace_expectations_with_statevector(qc, [("I", single_qubit_op)])

    with pytest.raises(ValueError, match="qargs has 1 entries"):
        trace_expectations_with_statevector(qc, [("ZZ", Operator(Pauli("ZZ")), [0])])


def test_expectation_trace_requires_power_of_two_dimension():
    qc = QuantumCircuit(1)
    qc.h(0)

    bad_matrix = np.eye(3)
    with pytest.raises(ValueError, match="power of two"):
        trace_expectations_with_statevector(qc, [("bad", Operator(bad_matrix))])


def test_expectation_trace_all_qubits_without_qargs():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    rows = trace_expectations_with_statevector(qc, [Operator(Pauli("ZZ"))])
    assert pytest.approx(rows[-1]["obs_0"]) == 1.0


def test_expectation_trace_rejects_complex_expectations():
    qc = QuantumCircuit(1)
    qc.h(0)

    non_hermitian = Operator(np.array([[1, 1j], [0, 1]], dtype=complex))
    with pytest.raises(ValueError, match="complex expectation"):
        trace_expectations_with_statevector(qc, [("bad", non_hermitian, [0])])


def test_expectation_trace_estimator_matches_statevector():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    observables = [
        ("ZZ", Pauli("ZZ"), [0, 1]),
        ("ZI", Pauli("ZI"), [0, 1]),
        ("X0", Pauli("X"), [0]),
    ]

    est = StatevectorEstimator()
    est_rows = trace_expectations_with_estimator(qc, observables, est, include_initial=True)
    exact_rows = trace_expectations_with_statevector(qc, observables, include_initial=True)

    assert len(est_rows) == len(exact_rows)
    for est_row, exact_row in zip(est_rows, exact_rows):
        assert est_row.keys() == exact_row.keys()
        for key in est_row:
            assert pytest.approx(est_row[key], abs=1e-12) == exact_row[key]


def test_expectation_trace_estimator_accepts_sparse_pauliop():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    observables = [
        ("ZZ", SparsePauliOp.from_list([("ZZ", 1.0)])),
        ("Z0", SparsePauliOp.from_list([("Z", 1.0)]), [0]),
    ]

    est = StatevectorEstimator()
    est_rows = trace_expectations_with_estimator(qc, observables, est, include_initial=True)
    exact_rows = trace_expectations_with_statevector(qc, observables, include_initial=True)

    assert len(est_rows) == len(exact_rows)
    for est_row, exact_row in zip(est_rows, exact_rows):
        assert est_row.keys() == exact_row.keys()
        for key in est_row:
            assert pytest.approx(est_row[key], abs=1e-12) == exact_row[key]


def test_expectation_trace_estimator_binds_parameters():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    est = StatevectorEstimator()
    params = {theta: np.pi / 2}

    rows_est = trace_expectations_with_estimator(qc, [Pauli("Z")], est, parameter_values=params)
    rows_exact = trace_expectations_with_statevector(qc, [Pauli("Z")], parameter_values=params)

    assert len(rows_est) == len(rows_exact)
    assert pytest.approx(rows_est[-1]["obs_0"], abs=1e-12) == rows_exact[-1]["obs_0"]


def test_expectation_trace_estimator_bindings_array():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    est = StatevectorEstimator()
    binding = BindingsArray({theta: np.pi / 2})

    rows_est = trace_expectations_with_estimator(
        qc,
        [Pauli("Z")],
        est,
        parameter_values=binding,
    )
    rows_exact = trace_expectations_with_statevector(
        qc,
        [Pauli("Z")],
        parameter_values={theta: np.pi / 2},
    )

    assert len(rows_est) == len(rows_exact)
    assert pytest.approx(rows_est[-1]["obs_0"], abs=1e-12) == rows_exact[-1]["obs_0"]


def test_expectation_trace_estimator_bindings_array_multidimensional():
    params = [Parameter("theta0"), Parameter("theta1")]
    qc = QuantumCircuit(1)
    for param in params:
        qc.ry(param, 0)

    est = StatevectorEstimator()
    values = np.array(
        [
            [[0.0, 0.0], [0.0, np.pi]],
            [[np.pi, 0.0], [np.pi, np.pi]],
        ]
    )
    binding = BindingsArray({tuple(params): values})

    rows_est = trace_expectations_with_estimator(
        qc,
        [Pauli("Z")],
        est,
        parameter_values=binding,
    )

    combos = values.reshape(-1, values.shape[-1])
    accum = 0.0
    for theta0, theta1 in combos:
        bound = qc.assign_parameters({params[0]: theta0, params[1]: theta1})
        state = Statevector.from_instruction(bound)
        accum += float(state.expectation_value(Pauli("Z")))
    expected = accum / combos.shape[0]

    assert pytest.approx(rows_est[-1]["obs_0"], abs=1e-12) == expected


def test_expectation_trace_estimator_bindings_array_missing_parameter():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.ry(phi, 0)

    est = StatevectorEstimator()
    binding = BindingsArray({theta: 0.0})

    with pytest.raises(
        ValueError,
        match="parameter_values are missing assignments for circuit parameters",
    ):
        trace_expectations_with_estimator(
            qc,
            [Pauli("Z")],
            est,
            parameter_values=binding,
        )


def test_expectation_trace_estimator_sequence_bindings():
    theta0 = Parameter("theta0")
    theta1 = Parameter("theta1")

    qc = QuantumCircuit(1)
    qc.ry(theta0, 0)
    qc.ry(theta1, 0)

    est = StatevectorEstimator()
    observables = [("Z", Pauli("Z"), [0])]

    bindings = [
        {theta0: np.pi / 4, theta1: 0.0},
        {theta0: np.pi / 4, theta1: np.pi / 2},
    ]

    rows = trace_expectations_with_estimator(qc, observables, est, parameter_values=bindings)

    assert len(rows) == 2
    assert pytest.approx(rows[0]["Z"], abs=1e-12) == np.cos(np.pi / 4)
    assert pytest.approx(rows[1]["Z"], abs=1e-12) == np.cos(np.pi / 4 + np.pi / 2)


def test_expectation_trace_estimator_list_bindings():
    theta0 = Parameter("theta0")
    theta1 = Parameter("theta1")

    qc = QuantumCircuit(1)
    qc.ry(theta0, 0)
    qc.ry(theta1, 0)

    est = StatevectorEstimator()

    observables = [Pauli("Z")]
    per_prefix = [[np.pi / 4, 0.0], [np.pi / 4, np.pi / 2]]

    rows = trace_expectations_with_estimator(qc, observables, est, parameter_values=per_prefix)

    assert len(rows) == 2
    assert pytest.approx(rows[0]["obs_0"], abs=1e-12) == np.cos(np.pi / 4)
    assert pytest.approx(rows[1]["obs_0"], abs=1e-12) == np.cos(np.pi / 4 + np.pi / 2)


def test_expectation_trace_estimator_iterable_bindings():
    theta = Parameter("theta")

    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.ry(theta, 0)

    est = StatevectorEstimator()
    observables = [("Z", Pauli("Z"), [0])]

    bindings = ({theta: val} for val in (0.0, np.pi / 2))
    rows = trace_expectations_with_estimator(qc, observables, est, parameter_values=bindings)

    assert len(rows) == 2
    assert pytest.approx(rows[0]["Z"], abs=1e-12) == 1.0
    assert pytest.approx(rows[1]["Z"], abs=1e-12) == -1.0


def test_expectation_trace_estimator_rejects_nonhermitian():
    qc = QuantumCircuit(1)
    qc.h(0)

    est = StatevectorEstimator()
    non_hermitian = Operator(np.array([[1, 1j], [0, 1]], dtype=complex))

    with pytest.raises(ValueError, match="complex coefficients"):
        trace_expectations_with_estimator(qc, [("bad", non_hermitian, [0])], est)


def test_expectation_trace_estimator_detects_complex_results():
    qc = QuantumCircuit(1)
    qc.h(0)

    fake_est = _FakeEstimator([1 + 5e-2j])

    with pytest.raises(ValueError, match="complex expectation values"):
        trace_expectations_with_estimator(qc, [Pauli("Z")], fake_est)

    tolerant_est = _FakeEstimator([1 + 1e-13j])
    rows = trace_expectations_with_estimator(qc, [Pauli("Z")], tolerant_est)
    assert rows == [{"obs_0": pytest.approx(1.0)}]


def test_expectation_trace_estimator_respects_relative_tolerance():
    qc = QuantumCircuit(1)
    qc.h(0)

    tolerant_est = _FakeEstimator([1e9 + 5e-4j])
    rows = trace_expectations_with_estimator(qc, [Pauli("Z")], tolerant_est)
    assert rows == [{"obs_0": pytest.approx(1e9)}]


def test_expectation_trace_estimator_validates_result_shape():
    qc = QuantumCircuit(1)
    qc.h(0)

    mismatch_est = _FakeEstimator([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="unexpected size"):
        trace_expectations_with_estimator(qc, [Pauli("X"), Pauli("Z")], mismatch_est)


def test_expectation_trace_estimator_rejects_measurements():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    est = StatevectorEstimator()

    with pytest.raises(
        ValueError,
        match="requires circuits without measurements",
    ):
        trace_expectations_with_estimator(qc, [Pauli("Z")], est)


def test_expectation_trace_surfaces_type_error(monkeypatch):
    """Type errors from qargs evaluation should not be masked by fallback calls."""

    qc = QuantumCircuit(1)
    qc.h(0)

    original_expectation = Statevector.expectation_value

    def fake_expectation(self, operator, **kwargs):
        if kwargs.get("qargs") is not None:
            raise TypeError("qargs unsupported in fake backend")
        return original_expectation(self, operator, **kwargs)

    monkeypatch.setattr(Statevector, "expectation_value", fake_expectation)

    with pytest.raises(TypeError, match="does not support the 'qargs' keyword"):
        trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), [0])])


def test_expectation_trace_detects_missing_qargs_keyword(monkeypatch):
    """Gracefully fail when expectation_value lacks a qargs parameter entirely."""

    qc = QuantumCircuit(1)
    qc.h(0)

    original_expectation = Statevector.expectation_value

    def fake_expectation(self, operator):
        return original_expectation(self, operator)

    monkeypatch.setattr(Statevector, "expectation_value", fake_expectation)

    with pytest.raises(TypeError, match="does not support the 'qargs' keyword"):
        trace_expectations_with_statevector(qc, [("Z", Pauli("Z"), [0])])
