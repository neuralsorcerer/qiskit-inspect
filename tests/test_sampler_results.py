from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers import BitArray, DataBin

from qiskit_inspect.sampler_results import extract_counts, extract_total_shots


def test_extract_counts_prefers_join_data():
    bitarray = BitArray.from_counts({"0": 3, "1": 2})

    class _PubResult:
        def join_data(self):
            return bitarray

    pub = _PubResult()

    assert extract_counts(pub) == {"0": 3, "1": 2}


def test_extract_counts_handles_join_data_iterable():
    bitarray = BitArray.from_counts({"1": 4})

    class _PubResult:
        def join_data(self):
            return (bitarray,)

    pub = _PubResult()

    assert extract_counts(pub) == {"1": 4}


def test_extract_counts_handles_join_data_mapping():
    bitarray = BitArray.from_counts({"01": 3})
    datab = DataBin(c=bitarray)

    class _PubResult:
        def join_data(self):
            return datab

    pub = _PubResult()

    assert extract_counts(pub) == {"1": 3}


def test_extract_counts_handles_nested_join_data_mapping():
    bitarray = BitArray.from_counts({"1": 6})
    datab = DataBin(c=bitarray)

    class _PubResult:
        def join_data(self):
            return [datab]

    pub = _PubResult()

    assert extract_counts(pub) == {"1": 6}


def test_extract_counts_handles_data_c_attribute():
    bitarray = BitArray.from_counts({"00": 1, "11": 4})
    data = SimpleNamespace(c=bitarray)
    pub = SimpleNamespace(data=data)

    assert extract_counts(pub) == {"00": 1, "11": 4}


def test_extract_counts_handles_data_iterable():
    bitarray = BitArray.from_counts({"10": 7})

    class _Data(list):
        pass

    data = _Data([bitarray])
    pub = SimpleNamespace(data=data)

    assert extract_counts(pub) == {"10": 7}


def test_extract_counts_handles_iterable_data_bin_values():
    bitarray = BitArray.from_counts({"0": 2, "1": 1})
    data = SimpleNamespace(values=lambda: (bitarray,))
    pub = SimpleNamespace(data=data)

    assert extract_counts(pub) == {"0": 2, "1": 1}


def test_extract_counts_handles_measurementless_circuit():
    sampler = StatevectorSampler()
    circuit = QuantumCircuit(1)
    with pytest.warns(UserWarning):
        result = sampler.run([circuit], shots=100).result()
    pub = result[0]

    assert extract_counts(pub) == {"": 100}


def test_extract_counts_uses_metadata_string_shots():
    pub = SimpleNamespace(metadata={"shots": "256"})

    assert extract_counts(pub) == {"": 256}


def test_extract_counts_uses_nested_metadata_shots():
    pub = SimpleNamespace(metadata={"shots": {"total": "42"}})

    assert extract_counts(pub) == {"": 42}


def test_extract_counts_uses_object_metadata_shots():
    class _ShotObj:
        def __init__(self, total):
            self.total = total

    pub = SimpleNamespace(metadata=SimpleNamespace(shots=_ShotObj(17)))

    assert extract_counts(pub) == {"": 17}


def test_extract_counts_sums_iterable_metadata_shots():
    pub = SimpleNamespace(metadata={"shots": [5, 7, 8]})

    assert extract_counts(pub) == {"": 20}


def test_extract_total_shots_handles_unknown_shapes():
    class _ShotObj:
        def __init__(self, total=None, shots=None):
            self.total = total
            self.shots = shots

    assert extract_total_shots({"shots": None}) is None
    assert extract_total_shots({}) is None
    assert extract_total_shots(SimpleNamespace(shots=_ShotObj(shots="13"))) == 13
    assert extract_total_shots(SimpleNamespace(shots=_ShotObj(total=9))) == 9
    assert extract_total_shots({"shots": (_ShotObj(total=2), _ShotObj(total=3))}) == 5


def test_extract_total_shots_accepts_top_level_metadata():
    metadata = SimpleNamespace(total="21")
    assert extract_total_shots(metadata) == 21


def test_extract_total_shots_handles_shot_count_key():
    assert extract_total_shots({"shot_count": "19"}) == 19
