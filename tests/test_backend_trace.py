from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Parameter
from qiskit.exceptions import QiskitError
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector

import qiskit_inspect.backend_trace as backend_trace
from qiskit_inspect import (
    trace_counts_with_sampler,
    trace_marginal_probabilities_with_sampler,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)
from qiskit_inspect.backend_trace import ObsSpec, _has_mid_circuit_measurement
from qiskit_inspect.prefix_builders import build_prefix_circuits


def test_prefix_count_matches_length():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    sampler = StatevectorSampler(default_shots=1024)
    probs = trace_probabilities_with_sampler(qc, sampler)
    assert len(probs) == len(qc.data)


def test_trace_counts_with_sampler_sums_to_shots():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=512)
    counts = trace_counts_with_sampler(qc, sampler, shots=512)

    assert len(counts) == len(qc.data)
    for entry in counts:
        assert sum(entry.values()) == 512
        assert all(isinstance(v, int) for v in entry.values())
        assert all(set(key) <= {"0", "1"} for key in entry)


def test_trace_counts_with_sampler_allows_backend_default_shots():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 3, "1": 1}

    class _RecordingSampler:
        def __init__(self, counts):
            self._counts = counts
            self.kwargs = []

        def run(self, pubs, **kwargs):
            self.kwargs.append(kwargs)
            results = []
            for _ in pubs:
                data = SimpleNamespace(get_counts=lambda: dict(self._counts))
                res = SimpleNamespace(join_data=lambda data=data: data, metadata=None)
                results.append(res)
            return SimpleNamespace(result=lambda results=results: results)

    sampler = _RecordingSampler(counts_payload)
    counts = trace_counts_with_sampler(qc, sampler, shots=None)

    assert sampler.kwargs == [{}]
    assert len(counts) == len(qc.data)
    assert counts[-1] == counts_payload


def test_trace_counts_with_sampler_uses_metadata_shots():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 30, "1": 20}

    class _DummyPubRes:
        def __init__(self, counts):
            self._counts = counts
            self.metadata = {"shots": 50}

        def join_data(self):
            class _Data:
                def __init__(self, counts):
                    self._counts = counts

                def get_counts(self):
                    return self._counts

            return _Data(self._counts)

    class _DummyJob:
        def __init__(self, results):
            self._results = results

        def result(self):
            return self._results

    class _DummySampler:
        def __init__(self, payload):
            self._payload = payload

        def run(self, pubs, shots=None):
            assert shots == 100
            results = [_DummyPubRes(self._payload) for _ in pubs]
            return _DummyJob(results)

    sampler = _DummySampler(counts_payload)
    counts = trace_counts_with_sampler(qc, sampler, shots=100)

    assert len(counts) == len(qc.data)
    final_counts = counts[-1]
    assert final_counts == counts_payload
    assert sum(final_counts.values()) == 50


def test_trace_counts_with_sampler_supports_object_shots_metadata():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 18, "1": 24}

    class _ShotObj:
        def __init__(self, total):
            self.total = total

    class _DummyPubRes:
        def __init__(self, counts, total):
            self._counts = counts
            self.metadata = SimpleNamespace(shots=_ShotObj(total))

        def join_data(self):
            class _Data:
                def __init__(self, counts):
                    self._counts = counts

                def get_counts(self):
                    return self._counts

            return _Data(self._counts)

    class _DummyJob:
        def __init__(self, results):
            self._results = results

        def result(self):
            return self._results

    class _DummySampler:
        def __init__(self, payload, total):
            self._payload = payload
            self._total = total

        def run(self, pubs, shots=None):
            assert shots == 100
            results = [_DummyPubRes(self._payload, self._total) for _ in pubs]
            return _DummyJob(results)

    sampler = _DummySampler(counts_payload, 42)
    counts = trace_counts_with_sampler(qc, sampler, shots=100)

    final_counts = counts[-1]
    assert final_counts == counts_payload
    assert sum(final_counts.values()) == 42


def test_trace_sampler_discards_unmeasured_classical_bits():
    qc = QuantumCircuit(1, 1)
    qc.h(0)

    sampler = StatevectorSampler(default_shots=4096)
    sampled_probs = trace_probabilities_with_sampler(qc, sampler, shots=4096)
    exact_probs = trace_probabilities_with_statevector_exact(qc)

    assert len(sampled_probs) == len(exact_probs)
    for sampled, exact in zip(sampled_probs, exact_probs):
        assert all(len(key) == 1 for key in sampled)
        assert set(sampled) == set(exact)
        assert sum(sampled.values()) == pytest.approx(1.0, abs=1e-9)

    sampled_counts = trace_counts_with_sampler(qc, sampler, shots=4096)
    assert len(sampled_counts) == len(sampled_probs)
    for entry in sampled_counts:
        assert all(len(key) == 1 for key in entry)
        assert sum(entry.values()) == 4096


def test_trace_probabilities_with_sampler_allows_backend_default_shots():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 7, "1": 5}

    class _RecordingSampler:
        def __init__(self, counts):
            self._counts = counts
            self.kwargs = []

        def run(self, pubs, **kwargs):
            self.kwargs.append(kwargs)
            results = []
            for _ in pubs:
                data = SimpleNamespace(get_counts=lambda: dict(self._counts))
                res = SimpleNamespace(join_data=lambda data=data: data, metadata=None)
                results.append(res)
            return SimpleNamespace(result=lambda results=results: results)

    sampler = _RecordingSampler(counts_payload)
    probs = trace_probabilities_with_sampler(qc, sampler, shots=None)

    assert sampler.kwargs == [{}]
    assert len(probs) == len(qc.data)
    final = probs[-1]
    total = sum(counts_payload.values())
    expected = {k: v / total for k, v in counts_payload.items()}
    assert final == pytest.approx(expected)


def test_trace_probabilities_with_sampler_validates_metadata_total():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 3, "1": 1}

    class _DummyPubRes:
        def __init__(self, counts, total):
            self._counts = counts
            self.metadata = {"shots": total}

        def join_data(self):
            class _Data:
                def __init__(self, counts):
                    self._counts = counts

                def get_counts(self):
                    return self._counts

            return _Data(self._counts)

    class _DummyJob:
        def __init__(self, results):
            self._results = results

        def result(self):
            return self._results

    class _DummySampler:
        def __init__(self, payload, declared_total):
            self._payload = payload
            self._declared_total = declared_total

        def run(self, pubs, shots=None):  # pylint: disable=unused-argument
            results = [_DummyPubRes(self._payload, self._declared_total) for _ in pubs]
            return _DummyJob(results)

    sampler = _DummySampler(counts_payload, declared_total=8)

    with pytest.raises(ValueError):
        trace_probabilities_with_sampler(qc, sampler, shots=100)


def test_trace_marginal_probabilities_with_sampler_validates_metadata_total():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    counts_payload = {"0": 5, "1": 3}

    class _DummyPubRes:
        def __init__(self, counts, total):
            self._counts = counts
            self.metadata = {"shots": total}

        def join_data(self):
            class _Data:
                def __init__(self, counts):
                    self._counts = counts

                def get_counts(self):
                    return self._counts

            return _Data(self._counts)

    class _DummyJob:
        def __init__(self, results):
            self._results = results

        def result(self):
            return self._results

    class _DummySampler:
        def __init__(self, payload, declared_total):
            self._payload = payload
            self._declared_total = declared_total

        def run(self, pubs, shots=None):  # pylint: disable=unused-argument
            results = [_DummyPubRes(self._payload, self._declared_total) for _ in pubs]
            return _DummyJob(results)

    sampler = _DummySampler(counts_payload, declared_total=16)

    with pytest.raises(ValueError):
        trace_marginal_probabilities_with_sampler(qc, sampler, [0], shots=100)


def test_trace_sampler_ignores_conditional_measurements():
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    conditional = Instruction("measure", 1, 1, [])
    conditional.condition = (qc.cregs[0], 1)
    qc.append(conditional, [qc.qubits[1]], [qc.clbits[0]])

    sampler = StatevectorSampler(default_shots=4096)
    sampled_probs = trace_probabilities_with_sampler(qc, sampler, shots=4096)
    exact_probs = trace_probabilities_with_statevector_exact(qc)

    assert len(sampled_probs) == len(exact_probs)
    for sampled, exact in zip(sampled_probs, exact_probs):
        expected_width = len(next(iter(exact))) if exact else 0
        assert all(len(key) == expected_width for key in sampled)
        assert sum(sampled.values()) == pytest.approx(1.0, abs=1e-9)

    sampled_counts = trace_counts_with_sampler(qc, sampler, shots=4096)
    assert len(sampled_counts) == len(sampled_probs)
    for entry, exact in zip(sampled_counts, exact_probs):
        expected_width = len(next(iter(exact))) if exact else 0
        assert all(len(key) == expected_width for key in entry)
        assert sum(entry.values()) == 4096


def test_trace_probabilities_with_sampler_parameter_bindings():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=256)
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=256,
        parameter_values={theta: 0.0},
    )

    assert len(probs) == len(qc.data)
    final = probs[-1]
    assert set(final) == {"0"}
    assert final["0"] == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_parameter_sequence():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=128)
    prefixes = len(qc.data)
    bindings = [{theta: 0.0} for _ in range(prefixes)]
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=128,
        parameter_values=bindings,
    )

    assert len(probs) == prefixes
    assert probs[-1]["0"] == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_validates_shots():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=256)

    with pytest.raises(ValueError):
        trace_probabilities_with_sampler(qc, sampler, shots=-1)

    with pytest.raises(TypeError):
        trace_probabilities_with_sampler(qc, sampler, shots=1.5)


def test_trace_probabilities_with_sampler_nested_parameter_binding():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=128)
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=128,
        parameter_values=[[0.0]],
    )

    assert len(probs) == len(qc.data)
    assert probs[-1]["0"] == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_scalar_bindings_per_prefix():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=2048)
    values = [0.0, np.pi]
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=2048,
        parameter_values=values,
    )

    assert len(probs) == len(qc.data)
    first = probs[0]
    assert len(first) == 1
    assert next(iter(first.values())) == pytest.approx(1.0, abs=1e-9)

    final_key, final_val = next(iter(probs[-1].items()))
    assert final_key.endswith("1")
    assert final_val == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_bindings_array_scalar():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=512)
    binding = BindingsArray({theta: 0.0})

    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=512,
        parameter_values=binding,
    )

    assert len(probs) == len(qc.data)
    assert probs[-1]["0"] == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_bindings_array_sequence():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=256)
    binding = BindingsArray({theta: [0.0, np.pi]})

    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=256,
        parameter_values=binding,
    )

    assert len(probs) == len(qc.data)
    final = probs[-1]
    assert pytest.approx(final["0"], abs=1e-9) == 0.5
    assert pytest.approx(final["1"], abs=1e-9) == 0.5


def test_trace_probabilities_with_sampler_bindings_array_multidimensional():
    theta0 = Parameter("theta0")
    theta1 = Parameter("theta1")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta0, 0)
    qc.ry(theta1, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=256)
    values = np.array(
        [
            [[0.0, 0.0], [0.0, np.pi]],
            [[np.pi, 0.0], [np.pi, np.pi]],
        ]
    )
    binding = BindingsArray({(theta0, theta1): values})

    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=256,
        parameter_values=binding,
    )

    assert len(probs) == len(qc.data)
    final = probs[-1]
    assert pytest.approx(final["0"], abs=1e-9) == 0.5
    assert pytest.approx(final["1"], abs=1e-9) == 0.5


def test_trace_probabilities_with_sampler_bindings_array_rejects_unknown_parameter():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=256)
    binding = BindingsArray({phi: 0.0})

    with pytest.raises(
        ValueError,
        match="parameter_values include an assignment for a parameter that is not present",
    ):
        trace_probabilities_with_sampler(
            qc,
            sampler,
            shots=256,
            parameter_values=binding,
        )


def test_trace_probabilities_with_sampler_bindings_array_missing_parameter():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.ry(phi, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=128)
    binding = BindingsArray({theta: 0.0})

    with pytest.raises(
        ValueError,
        match="parameter_values are missing assignments for circuit parameters",
    ):
        trace_probabilities_with_sampler(
            qc,
            sampler,
            shots=128,
            parameter_values=binding,
        )


def test_sampler_parameter_values_iterable_consumed_per_prefix():
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    prefixes = backend_trace._prefixes_with_end_measure(qc)
    values = ({theta: angle} for angle in (0.0, np.pi / 2, np.pi))
    pubs = backend_trace._build_sampler_pubs(prefixes, values, (theta,))

    assert len(pubs) == len(prefixes)
    bindings = [binding for _, binding in pubs]
    assert [binding[theta] for binding in bindings] == [0.0, np.pi / 2, np.pi]


def test_trace_probabilities_with_sampler_multi_parameter_sequence():
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    qc = QuantumCircuit(1, 1)
    qc.ry(alpha, 0)
    qc.rz(beta, 0)
    qc.measure(0, 0)

    sampler = StatevectorSampler(default_shots=512)
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=512,
        parameter_values=[np.pi, 0.0],
    )

    assert len(probs) == len(qc.data)
    final = probs[-1]
    assert "1" in final
    assert final["1"] == pytest.approx(1.0, abs=1e-9)


def test_trace_probabilities_with_sampler_parameter_mapping_filtered_to_prefix():
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    qc = QuantumCircuit(2, 2)
    qc.ry(alpha, 0)
    qc.cx(0, 1)
    qc.rz(beta, 1)
    qc.measure([0, 1], [0, 1])

    sampler = StatevectorSampler(default_shots=256)
    probs = trace_probabilities_with_sampler(
        qc,
        sampler,
        shots=256,
        parameter_values={alpha: 0.0, beta: 0.0},
    )

    assert len(probs) == len(qc.data)
    final = probs[-1]
    assert set(final) == {"00"}
    assert final["00"] == pytest.approx(1.0, abs=1e-9)


def test_conditional_measurement_triggers_scratch_measure():
    qc = QuantumCircuit(1, 1)
    meas_inst = Instruction("measure", 1, 1, [])
    meas_inst.condition = (qc.cregs[0], 1)
    qc.append(meas_inst, [qc.qubits[0]], [qc.clbits[0]])

    prefixes = build_prefix_circuits(qc)
    assert len(prefixes) == 1
    measures = [ci for ci in prefixes[0].data if ci.operation.name == "measure"]
    assert len(measures) == 2  # conditional + scratch measurement
    scratch_regs = [reg for reg in prefixes[0].cregs if reg.name.startswith("extra_m")]
    assert scratch_regs, "Expected scratch register for conditional measurement"


def test_remeasure_when_classical_bit_reused():
    """If a classical bit is reused, the original qubit should be remeasured."""

    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.x(1)
    qc.measure(1, 0)

    sampler = StatevectorSampler(default_shots=256)
    probs = trace_probabilities_with_sampler(qc, sampler, shots=256)

    # Final prefix should contain both qubit outcomes (q0=0, q1=1 -> "01").
    last = probs[-1]
    assert set(last) == {"01"}
    assert last["01"] == pytest.approx(1.0, abs=1e-9)


def test_conditional_overwrite_remeasures_previous_qubit():
    """Conditional measurements that reuse bits should keep earlier qubit outcomes."""

    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.measure(0, 0)
    cond = Instruction("measure", 1, 1, [])
    cond.condition = (qc.cregs[0], 0)
    qc.append(cond, [qc.qubits[1]], [qc.clbits[0]])

    prefixes = build_prefix_circuits(qc)
    final_pref = prefixes[-1]
    scratch_regs = [reg for reg in final_pref.cregs if reg.name.startswith("extra_m")]
    assert scratch_regs, "Expected scratch measurements for overwritten qubits"
    scratch_bits = {bit for reg in scratch_regs for bit in reg}

    remeasured_qubits = set()
    for ci in final_pref.data:
        if ci.operation.name != "measure":
            continue
        for qbit, cbit in zip(ci.qubits, ci.clbits):
            if cbit in scratch_bits:
                remeasured_qubits.add(final_pref.find_bit(qbit).index)

    # Both qubits should be measured into scratch bits: q0 from the earlier measurement
    # and q1 because the conditional measurement may not execute.
    assert remeasured_qubits == {0, 1}

    sampler = StatevectorSampler(default_shots=1024)
    probs = trace_probabilities_with_sampler(qc, sampler, shots=1024)
    bitstring, value = next(iter(probs[-1].items()))
    assert value == pytest.approx(1.0, abs=1e-9)

    # Map scratch bits back to their classical indices and confirm the stored values.
    bit_indices = {final_pref.find_bit(c).index for c in scratch_bits}
    assert len(bit_indices) == 2

    def bit_at(s: str, idx: int) -> str:
        pos = len(s) - 1 - idx
        return s[pos]

    # One of the scratch positions corresponds to q0 and should be 0; the other to q1 and should be 1.
    observed_values = {bit_at(bitstring, idx) for idx in bit_indices}
    assert observed_values == {"0", "1"}


def test_trace_probabilities_with_sampler_normalizes_keys(monkeypatch):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    class _FakeJob:
        def __init__(self, count: int) -> None:
            self._count = count

        def result(self):
            return [object()] * self._count

    class _FakeSampler:
        def run(self, circuits, shots: int):
            assert shots == 4
            return _FakeJob(len(circuits))

    def fake_extract_counts(_pub_res):
        return {np.str_("1 "): 3, (0,): 1}

    monkeypatch.setattr("qiskit_inspect.backend_trace.extract_counts", fake_extract_counts)

    sampler = _FakeSampler()
    probs = trace_probabilities_with_sampler(qc, sampler, shots=4)

    assert probs
    assert any("1" in entry for entry in probs)
    for entry in probs:
        assert all(isinstance(k, str) for k in entry)
        assert all(set(k) <= {"0", "1"} for k in entry)
        assert sum(entry.values()) == pytest.approx(1.0)
        if "1" in entry:
            assert entry["1"] == pytest.approx(0.75)
        if "0" in entry and len(entry) > 1:
            assert entry["0"] == pytest.approx(0.25)


def test_trace_counts_with_sampler_normalizes_keys(monkeypatch):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    class _FakeJob:
        def __init__(self, count: int) -> None:
            self._count = count

        def result(self):
            return [object()] * self._count

    class _FakeSampler:
        def run(self, circuits, shots: int):
            assert shots == 4
            return _FakeJob(len(circuits))

    def fake_extract_counts(_pub_res):
        return {np.str_("1 "): 3, (0,): 1}

    monkeypatch.setattr("qiskit_inspect.backend_trace.extract_counts", fake_extract_counts)

    sampler = _FakeSampler()
    counts = trace_counts_with_sampler(qc, sampler, shots=4)

    non_empty = [entry for entry in counts if entry]
    assert non_empty
    assert any("1" in entry for entry in non_empty)
    for entry in non_empty:
        assert all(set(key) <= {"0", "1"} for key in entry)
        assert sum(entry.values()) == 4
        if "1" in entry:
            assert entry["1"] == 3
        if "0" in entry and len(entry) > 1:
            assert entry["0"] == 1


def test_trace_counts_with_sampler_handles_large_counts(monkeypatch):
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    large = 2**60 + 3

    class _FakeJob:
        def __init__(self, count: int) -> None:
            self._count = count

        def result(self):
            return [object()] * self._count

    class _FakeSampler:
        def run(self, circuits, shots: int):
            assert shots == large
            return _FakeJob(len(circuits))

    def fake_extract_counts(_pub_res):
        return {"1": large}

    monkeypatch.setattr("qiskit_inspect.backend_trace.extract_counts", fake_extract_counts)

    sampler = _FakeSampler()
    counts = trace_counts_with_sampler(qc, sampler, shots=large)

    assert counts
    for entry in counts:
        if not entry:
            continue
        assert entry["1"] == large


def test_normalize_counts_dict_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        backend_trace._normalize_counts_dict({"1": -1}, width=1)


def test_normalize_counts_dict_rejects_non_integral_values():
    with pytest.raises(TypeError, match="integer value"):
        backend_trace._normalize_counts_dict({"1": 1.5}, width=1)


def test_normalize_counts_dict_rejects_fractional_numeric_types():
    with pytest.raises(TypeError, match="integer value"):
        backend_trace._normalize_counts_dict({"1": Decimal("1.2")}, width=1)
    with pytest.raises(TypeError, match="integer value"):
        backend_trace._normalize_counts_dict({"1": Fraction(3, 2)}, width=1)


def test_normalize_counts_dict_accepts_other_integral_numeric_types():
    result = backend_trace._normalize_counts_dict({"1": Decimal("2"), "0": Fraction(3, 1)}, width=1)
    assert result == {"0": 3, "1": 2}


def test_normalize_counts_dict_validates_total_shots():
    with pytest.raises(ValueError, match="total_shots"):
        backend_trace._normalize_counts_dict({}, width=1, total_shots=5)


def test_normalize_counts_dict_accepts_integral_total_shots_variants():
    counts = {"0": 3, "1": 2}
    expected = {"00": 3, "01": 2}

    for total_shots in (
        "5",
        Fraction(10, 2),
        Decimal("5"),
        np.int64(5),
        np.float64(5.0),
    ):
        result = backend_trace._normalize_counts_dict(counts, width=2, total_shots=total_shots)
        assert result == expected

    result = backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots=True)
    assert result == {"0": 1}


def test_normalize_counts_dict_rejects_non_integral_total_shots():
    with pytest.raises(TypeError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots=3.5)
    with pytest.raises(TypeError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots="3.5")
    with pytest.raises(TypeError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots=Fraction(7, 2))


def test_normalize_counts_dict_rejects_invalid_total_shots_inputs():
    with pytest.raises(TypeError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots="five")

    with pytest.raises(ValueError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots=Decimal("-1"))

    with pytest.raises(ValueError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 1}, width=1, total_shots=-2)


def test_normalize_counts_dict_rejects_mismatched_total_shots():
    with pytest.raises(ValueError, match="total_shots"):
        backend_trace._normalize_counts_dict({"0": 2}, width=1, total_shots=1)


def test_normalize_counts_dict_uses_total_shots_for_empty_zero_width():
    result = backend_trace._normalize_counts_dict({}, width=0, total_shots=7)
    assert result == {"": 7}


def test_sampler_mid_circuit_measurements_raise_runtime_error():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)

    sampler = StatevectorSampler(default_shots=256)

    with pytest.raises(RuntimeError, match="StatevectorSampler"):
        trace_probabilities_with_sampler(qc, sampler)

    with pytest.raises(RuntimeError, match="StatevectorSampler"):
        trace_marginal_probabilities_with_sampler(qc, sampler, [0], add_measure_for_qubits=False)

    with pytest.raises(RuntimeError, match="StatevectorSampler"):
        trace_counts_with_sampler(qc, sampler)


def test_non_statevector_sampler_mid_measurement_error_message():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)

    class _FailingSampler:
        def run(self, _circuits, shots):  # pragma: no cover - intentionally fails
            raise QiskitError("Sampler cannot run mid circuit measurement helpers.")

    sampler = _FailingSampler()

    with pytest.raises(RuntimeError) as excinfo:
        trace_probabilities_with_sampler(qc, sampler)

    message = str(excinfo.value)
    assert "trace_probabilities_with_sampler" in message
    assert "trace_probabilities_with_statevector_exact" in message

    with pytest.raises(RuntimeError) as excinfo:
        trace_counts_with_sampler(qc, sampler)

    message = str(excinfo.value)
    assert "trace_counts_with_sampler" in message
    assert "trace_probabilities_with_statevector_exact" in message


def test_non_statevector_sampler_mid_measurement_error_message_marginals():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)

    class _FailingSampler:
        def run(self, _circuits, shots):  # pragma: no cover - intentionally fails
            raise QiskitError("Backend lacks mid-circuit measurement support")

    sampler = _FailingSampler()

    with pytest.raises(RuntimeError) as excinfo:
        trace_marginal_probabilities_with_sampler(qc, sampler, [0])

    message = str(excinfo.value)
    assert "trace_marginal_probabilities_with_sampler" in message
    assert "trace_marginal_probabilities_with_statevector" in message


def test_statevector_sampler_mid_measurement_detected_before_run(monkeypatch):
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)

    sampler = StatevectorSampler(default_shots=128)

    calls = {"run": 0}

    def _fail_run(*args, **kwargs):  # pragma: no cover - should not execute
        calls["run"] += 1
        raise AssertionError("StatevectorSampler.run should not be invoked")

    monkeypatch.setattr(sampler, "run", _fail_run)

    with pytest.raises(RuntimeError, match="StatevectorSampler"):
        trace_probabilities_with_sampler(qc, sampler)

    assert calls["run"] == 0


def test_statevector_sampler_mid_measure_detection_helper():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    assert not _has_mid_circuit_measurement(qc)

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.barrier()
    assert not _has_mid_circuit_measurement(qc)

    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.x(1)
    assert not _has_mid_circuit_measurement(qc)

    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    assert _has_mid_circuit_measurement(qc)

    qc = QuantumCircuit(1, 1)
    then_block = QuantumCircuit(1, 1)
    then_block.measure(0, 0)
    then_block.x(0)
    else_block = QuantumCircuit(1, 1)
    qc.if_else((qc.clbits[0], 0), then_block, else_block, [0], [0])
    assert _has_mid_circuit_measurement(qc)

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    then_block = QuantumCircuit(1, 1)
    then_block.measure(0, 0)
    else_block = QuantumCircuit(1, 1)
    qc.if_else((qc.clbits[0], 0), then_block, else_block, [0], [0])
    qc.x(0)
    assert _has_mid_circuit_measurement(qc)


def test_if_else_branch_measurement_without_reuse_is_not_flagged():
    qc = QuantumCircuit(1, 1)
    then_block = QuantumCircuit(1, 1)
    then_block.measure(0, 0)
    else_block = QuantumCircuit(1, 1)
    else_block.x(0)
    qc.if_else((qc.clbits[0], 0), then_block, else_block, [0], [0])
    assert not _has_mid_circuit_measurement(qc)


def test_loop_with_mid_circuit_measurement_is_detected():
    qc = QuantumCircuit(1, 1)
    body = QuantumCircuit(1, 1)
    body.x(0)
    body.measure(0, 0)
    qc.while_loop((qc.clbits[0], 0), body, [0], [0])
    assert _has_mid_circuit_measurement(qc)


def test_loop_measure_only_does_not_trigger_detection():
    qc = QuantumCircuit(1, 1)
    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    qc.for_loop(range(2), None, body, [0], [0])
    assert not _has_mid_circuit_measurement(qc)


def test_loop_measurement_after_classical_update_is_detected():
    qc = QuantumCircuit(1, 1)
    body = QuantumCircuit(1, 1)
    then_block = QuantumCircuit(1, 1)
    then_block.store(then_block.clbits[0], True)
    else_block = QuantumCircuit(1, 1)
    else_block.measure(0, 0)
    body.if_else((body.clbits[0], False), then_block, else_block, [0], [0])
    qc.for_loop(range(2), None, body, [0], [0])
    qc.x(0)
    assert _has_mid_circuit_measurement(qc)


def test_sparse_identity_cached_instances():
    first = backend_trace._sparse_identity(3)
    second = backend_trace._sparse_identity(3)

    assert first is second
    assert first.paulis.to_labels() == ["III"]
    assert first.coeffs.tolist() == [1.0]


def test_spec_to_sparse_pauli_qargs_embedding():
    op = Operator(Pauli("XZ"))

    spec = ObsSpec("obs", op, (0, 2))
    embedded = backend_trace._spec_to_sparse_pauli(spec, 4)

    assert embedded.num_qubits == 4
    assert embedded.paulis.to_labels() == ["IXIZ"]
    assert embedded.coeffs.tolist() == [1.0]


def test_spec_to_sparse_pauli_qargs_mismatch_raises():
    op = Operator(Pauli("X"))

    with pytest.raises(ValueError):
        backend_trace._spec_to_sparse_pauli(ObsSpec("obs", op, (0, 1)), 2)
