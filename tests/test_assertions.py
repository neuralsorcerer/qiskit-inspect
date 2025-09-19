from qiskit.quantum_info import Statevector

from qiskit_inspect import assert_probabilities_close, assert_state_equiv
from qiskit_inspect.visual import ascii_histogram


def test_state_equiv():
    a = Statevector.from_label("+0")
    b = Statevector.from_label("+0")
    assert_state_equiv(a, b)


def test_prob_l1():
    assert_probabilities_close({"00": 0.5, "11": 0.5}, {"00": 0.49, "11": 0.51}, tol_l1=0.05)


def test_ascii_histogram_basic():
    txt = ascii_histogram({"0": 0.7, "1": 0.3}, width=10)
    assert "0:" in txt and "1:" in txt
    # ensure longer bar for the larger probability
    lines = {line.split(":")[0].strip(): line for line in txt.splitlines()}
    assert lines["0"].count("#") > lines["1"].count("#")
