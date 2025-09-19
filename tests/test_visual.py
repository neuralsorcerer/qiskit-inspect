import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_inspect.visual import pretty_ket


def test_pretty_ket_handles_negative_real_coefficients():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.z(0)
    state = Statevector.from_instruction(qc)

    assert pretty_ket(state) == "0.7071|0> - 0.7071|1>"


def test_pretty_ket_suppresses_zero_real_for_imaginary_terms():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    state = Statevector.from_instruction(qc)

    assert pretty_ket(state) == "0.7071|0> + 0.7071j|1>"


def test_pretty_ket_handles_negative_imaginary_terms():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.sdg(0)
    state = Statevector.from_instruction(qc)

    assert pretty_ket(state) == "0.7071|0> - 0.7071j|1>"


def test_pretty_ket_first_term_negative():
    state = Statevector([-1.0, 0.0])

    assert pretty_ket(state) == "-1.0000|0>"


def test_pretty_ket_preserves_small_components_above_threshold():
    amp = 0.0009 + 0.0009j
    dominant = math.sqrt(1 - abs(amp) ** 2)
    state = Statevector([dominant, amp])

    assert pretty_ket(state, threshold=1e-3).endswith("0.0009+0.0009j|1>")


def test_pretty_ket_zero_qubit_state_uses_empty_label():
    state = Statevector([1.0])

    assert pretty_ket(state) == "1.0000|>"
