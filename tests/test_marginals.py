import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.controlflow import ForLoopOp, WhileLoopOp
from qiskit.primitives import StatevectorSampler

import qiskit_inspect.backend_trace as backend_trace
from qiskit_inspect.backend_trace import trace_marginal_probabilities_with_sampler


class _FakeCountsResult:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class _FakeCountsJob:
    def __init__(self, counts_list):
        self._counts_list = counts_list

    def result(self):
        return [_FakeCountsResult(c) for c in self._counts_list]


class _FakeSampler:
    def __init__(self, counts_list):
        self._counts_list = counts_list

    def run(self, circuits, shots):
        assert len(circuits) == len(self._counts_list)
        return _FakeCountsJob(self._counts_list)


def _measurement_map(circ: QuantumCircuit) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for ci in circ.data:
        if ci.operation.name != "measure":
            continue
        if not ci.qubits or not ci.clbits:
            continue
        for qbit, cbit in zip(ci.qubits, ci.clbits):
            q = circ.find_bit(qbit).index
            c = circ.find_bit(cbit).index
            mapping[q] = c
    return mapping


def test_marginals_exist_each_prefix_selected_qubits():
    # Circuit with 2 qubits, but only measure one explicitly at end.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # No explicit measure in the middle
    qc.measure_all()

    sampler = StatevectorSampler(default_shots=512)
    # Request marginals on qubit 0 across all prefixes
    probs_list = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[0], shots=512, add_measure_for_qubits=True
    )

    # There are as many prefixes as operations
    assert len(probs_list) == len(qc.data)
    # Each prefix should have defined marginals for the requested qubit
    for p in probs_list:
        assert isinstance(p, dict)
        assert p  # non-empty
        # Probabilities should sum close to 1
        s = sum(p.values())
        assert abs(s - 1.0) < 1e-6


def test_trace_marginals_sampler_rejects_duplicate_qubits():
    qc = QuantumCircuit(1)
    sampler = StatevectorSampler(default_shots=16)
    with pytest.raises(ValueError, match="Duplicate qubit index"):
        trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[0, 0])


def test_trace_marginals_sampler_preserves_qubit_order_in_keys():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure_all()

    sampler = StatevectorSampler(default_shots=128)
    probs_list = trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[2, 0], shots=128)
    assert probs_list[-1] == {"01": 1.0}


def test_trace_marginals_sampler_requires_all_requested_qubits_measured(monkeypatch):
    """If a requested qubit lacks a measurement, the marginal should be empty."""

    def fake_prefixes(_circ):
        pref = QuantumCircuit(2, 2)
        pref.measure(0, 0)
        return [pref]

    monkeypatch.setattr(backend_trace, "_prefixes_with_end_measure", fake_prefixes)

    qc = QuantumCircuit(2)
    sampler = _FakeSampler([{"0": 6, "1": 2}])
    probs = trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[0, 1], shots=8)
    assert probs == [{}]


def test_trace_marginals_sampler_handles_multi_qubit_measure():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    meas_inst = Instruction("measure", 2, 2, [])
    qc.append(meas_inst, [qc.qubits[0], qc.qubits[1]], [qc.clbits[0], qc.clbits[1]])

    counts = [
        {"0000": 16},
        {"0000": 16},
        {"00": 8, "11": 8},
    ]
    sampler = _FakeSampler(counts)

    probs = trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[0, 1], shots=16)

    assert len(probs) == len(qc.data)
    assert probs[0] == {"00": 1.0}
    assert probs[1] == {"00": 1.0}
    assert probs[2] == {"00": 0.5, "11": 0.5}


def test_trace_marginals_sampler_empty_qubits_returns_trivial_distribution():
    qc = QuantumCircuit(1)
    qc.h(0)
    sampler = StatevectorSampler(default_shots=32)
    probs = trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[], shots=32)
    assert probs == [{"": 1.0} for _ in range(len(qc.data))]


def test_trace_marginals_sampler_empty_qubits_skips_sampler_run():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)

    class _FailingSampler:
        def run(self, _circuits, _shots):
            raise AssertionError("sampler.run should not be called when qubits is empty")

    sampler = _FailingSampler()
    probs = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[], shots=8, add_measure_for_qubits=True
    )
    assert probs == [{"": 1.0} for _ in range(len(qc.data))]


def test_prefixes_remeasure_qubits_after_post_measure_ops():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.reset(0)
    qc.x(0)

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    assert len(prefixes) == len(qc.data)

    first_map = _measurement_map(prefixes[0])
    assert first_map[0] == 0  # original classical bit

    final_pref = prefixes[-1]
    final_map = _measurement_map(final_pref)
    assert final_map[0] == qc.num_clbits  # remapped to scratch register
    assert any(reg.name.startswith("extra_m") for reg in final_pref.cregs)


def test_prefixes_selected_qubits_remeasure_after_post_measure_ops():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.reset(0)
    qc.x(0)

    prefixes = backend_trace._prefixes_with_end_measure_of_qubits(qc, [0])
    assert len(prefixes) == len(qc.data)

    final_pref = prefixes[-1]
    final_map = _measurement_map(final_pref)
    assert final_map[0] == qc.num_clbits
    assert any(reg.name.startswith("marg_m") for reg in final_pref.cregs)


def test_prefixes_ifelse_overwrite_remeasures_original_qubit():
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 0)):
        qc.measure(1, 0)

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    final_pref = prefixes[-1]

    scratch_regs = [reg for reg in final_pref.cregs if reg.name.startswith("extra_m")]
    assert scratch_regs, "Expected scratch measurements for overwritten classical bits"

    scratch_indices = {final_pref.find_bit(bit).index for reg in scratch_regs for bit in reg}

    qubit_to_clbits: dict[int, set[int]] = {}
    for inst in final_pref.data:
        if inst.operation.name != "measure":
            continue
        for qbit, cbit in zip(inst.qubits, inst.clbits):
            q_index = final_pref.find_bit(qbit).index
            c_index = final_pref.find_bit(cbit).index
            qubit_to_clbits.setdefault(q_index, set()).add(c_index)

    assert qubit_to_clbits[0] & scratch_indices
    assert qubit_to_clbits[1] & scratch_indices


def test_trace_marginals_sampler_uses_latest_measurement_after_reset():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.reset(0)
    qc.x(0)

    counts = [{"0": 1}, {"00": 1}, {"10": 1}]
    sampler = _FakeSampler(counts)
    probs = trace_marginal_probabilities_with_sampler(qc, sampler, qubits=[0], shots=1)

    assert len(probs) == len(qc.data)
    assert probs[-1] == {"1": 1.0}


def test_trace_marginals_sampler_handles_ifelse_overwrite(monkeypatch):
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 0)):
        qc.measure(1, 0)

    # Fake sampler outputs counts with the classical register overwritten to ``1``
    # while the scratch measurement for qubit 0 reports ``0``. Correct behaviour
    # should rely on the scratch bit, not the potentially overwritten classical bit.
    counts = [{"00": 4}, {"100": 4}]
    sampler = _FakeSampler(counts)

    probs = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[0], shots=4, add_measure_for_qubits=True
    )

    assert len(probs) == len(qc.data)
    assert probs[-1] == {"0": 1.0}


def test_prefixes_remeasure_after_classical_write():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    write = Instruction("write", 0, 1, [])
    qc.append(write, [], [qc.clbits[0]])

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    assert len(prefixes) == len(qc.data)

    final_map = _measurement_map(prefixes[-1])
    assert final_map[0] >= qc.num_clbits


def test_trace_marginals_sampler_uses_scratch_after_classical_write():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    write = Instruction("write", 0, 1, [])
    qc.append(write, [], [qc.clbits[0]])

    counts = [{"0": 4}, {"00": 4}]
    sampler = _FakeSampler(counts)

    probs = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[0], shots=4, add_measure_for_qubits=True
    )

    assert len(probs) == len(qc.data)
    assert probs[-1] == {"0": 1.0}


def test_prefixes_for_loop_overwrite_remeasures_original_qubit():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    loop = ForLoopOp((0, 1), None, body)
    qc.append(loop, [qc.qubits[0]], [qc.clbits[0]])

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    final_map = _measurement_map(prefixes[-1])

    assert final_map[0] >= qc.num_clbits


def test_trace_marginals_sampler_handles_for_loop_overwrite():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    loop = ForLoopOp((0, 1), None, body)
    qc.append(loop, [qc.qubits[0]], [qc.clbits[0]])

    counts = [{"0": 4}, {"01": 4}]
    sampler = _FakeSampler(counts)

    probs = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[0], shots=4, add_measure_for_qubits=True
    )

    assert len(probs) == len(qc.data)
    assert probs[-1] == {"0": 1.0}


def test_prefixes_while_loop_overwrite_remeasures_original_qubit():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    loop = WhileLoopOp((body.clbits[0], 0), body)
    qc.append(loop, [qc.qubits[0]], [qc.clbits[0]])

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    final_map = _measurement_map(prefixes[-1])

    assert final_map[0] >= qc.num_clbits


def test_trace_marginals_sampler_handles_while_loop_overwrite():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    loop = WhileLoopOp((body.clbits[0], 0), body)
    qc.append(loop, [qc.qubits[0]], [qc.clbits[0]])

    counts = [{"0": 4}, {"01": 4}]
    sampler = _FakeSampler(counts)

    probs = trace_marginal_probabilities_with_sampler(
        qc, sampler, qubits=[0], shots=4, add_measure_for_qubits=True
    )

    assert len(probs) == len(qc.data)
    assert probs[-1] == {"0": 1.0}
