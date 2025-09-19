from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qiskit_inspect.backend_trace import trace_probabilities_with_statevector_exact
from qiskit_inspect.probabilities import normalize_probability_dict


def test_normalize_probability_dict_mixed_key_types():
    raw = {
        np.str_("0 1"): np.float64(0.125),
        b"11": 0.125,
        (1, 0, 1): 0.25,
        (np.bool_(0), np.bool_(0), np.bool_(0)): 0.25,
        6: 0.25,
    }
    normalized = normalize_probability_dict(raw, num_qubits=3)
    assert normalized == {
        "001": 0.125,
        "011": 0.125,
        "101": 0.25,
        "000": 0.25,
        "110": 0.25,
    }


def test_normalize_probability_dict_preserves_tiny_values():
    tiny = 1e-16
    norm = normalize_probability_dict({"1": tiny}, num_qubits=1)
    assert norm["1"] == pytest.approx(tiny)


def test_normalize_probability_dict_clamps_small_negative_noise():
    norm = normalize_probability_dict({"0": -1e-16, "1": 1.0}, num_qubits=1)
    assert norm["0"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_normalize_probability_dict_rejects_non_finite_values(bad):
    with pytest.raises(ValueError):
        normalize_probability_dict({"0": bad}, num_qubits=1)


def test_normalize_probability_dict_rejects_large_negative():
    with pytest.raises(ValueError):
        normalize_probability_dict({"0": -1e-3}, num_qubits=1)


def test_normalize_probability_dict_handles_zero_qubits():
    assert normalize_probability_dict({0: 1.0}, num_qubits=0) == {"": 1.0}
    with pytest.raises(ValueError):
        normalize_probability_dict({1: 1.0}, num_qubits=0)


def test_normalize_probability_dict_zero_qubit_empty_mapping():
    assert normalize_probability_dict({}, num_qubits=0) == {"": 1.0}


def test_normalize_probability_dict_rejects_non_binary_strings():
    with pytest.raises(ValueError):
        normalize_probability_dict({"2": 0.5}, num_qubits=1)


def test_normalize_probability_dict_sequence_width_validation():
    with pytest.raises(ValueError):
        normalize_probability_dict({(1, 1, 0, 1): 1.0}, num_qubits=2)


def test_normalize_probability_dict_rejects_non_integral_num_qubits():
    with pytest.raises(TypeError):
        normalize_probability_dict({"0": 1.0}, num_qubits=2.0)
    with pytest.raises(TypeError):
        normalize_probability_dict({"0": 1.0}, num_qubits=np.float64(2.0))


def test_normalize_probability_dict_accepts_numpy_boolean_num_qubits():
    norm = normalize_probability_dict({"1": 1.0}, num_qubits=np.bool_(True))
    assert norm == {"1": 1.0}


def test_normalize_probability_dict_accepts_numpy_boolean_key():
    norm = normalize_probability_dict({np.bool_(True): 1.0})
    assert norm == {"1": 1.0}


def test_normalize_probability_dict_left_pads_strings():
    norm = normalize_probability_dict({"1": 0.5}, num_qubits=3)
    assert norm == {"001": 0.5}


def test_normalize_probability_dict_accumulates_colliding_keys():
    raw = {"1": 0.125, "01": 0.375, "001": 0.5}
    norm = normalize_probability_dict(raw)
    assert norm == {"001": 1.0}


def test_normalize_probability_dict_rejects_non_numeric_values():
    with pytest.raises(TypeError):
        normalize_probability_dict({"0": complex(0.1, 0.2)})


def test_normalize_probability_dict_orders_keys_lexicographically():
    raw = {"11": 0.25, "0": 0.5, "01": 0.25}
    norm = normalize_probability_dict(raw, num_qubits=2)
    assert list(norm) == ["00", "01", "11"]


def test_normalize_probability_dict_falls_back_to_string_coercion():
    class Weird:
        def __str__(self) -> str:
            return "1_0"

    raw = {Weird(): 0.5, " 0 ": 0.5}
    norm = normalize_probability_dict(raw, num_qubits=2)
    assert norm == {"10": 0.5, "00": 0.5}


def test_normalize_probability_dict_accepts_supports_index_keys():
    class Indexable:
        def __init__(self, value: int) -> None:
            self._value = value

        def __index__(self) -> int:
            return self._value

    raw = {Indexable(3): 0.5}
    norm = normalize_probability_dict(raw)
    assert norm == {"11": 0.5}


def test_normalize_probability_dict_accepts_supports_index_num_qubits():
    class Width:
        def __index__(self) -> int:
            return 3

    norm = normalize_probability_dict({"1": 1.0}, num_qubits=Width())
    assert norm == {"001": 1.0}


def test_trace_probabilities_statevector_exact_handles_zero_qubit_circuit():
    qc = QuantumCircuit()
    assert trace_probabilities_with_statevector_exact(qc) == []
    assert trace_probabilities_with_statevector_exact(qc, include_initial=True) == [{"": 1.0}]
