import pytest
from qiskit import QuantumCircuit

pytest.importorskip("pandas")

from qiskit_inspect import (
    CircuitDebugger,
    counts_to_dataframe,
    expectations_to_dataframe,
    probabilities_to_dataframe,
    trace_records_to_dataframe,
)


def _simple_circuit():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _deterministic_circuit():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    return qc


def test_trace_records_dataframe_probabilities():
    dbg = CircuitDebugger(_simple_circuit())
    records = dbg.trace(include_initial=True)

    df = trace_records_to_dataframe(records, state_format="probs")

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["step_index", "instruction", "classical_bits", "p_0", "p_1"]
    assert df.iloc[0]["p_0"] == pytest.approx(1.0)
    assert df.iloc[1]["p_0"] == pytest.approx(0.5)
    assert df.iloc[1]["p_1"] == pytest.approx(0.5)
    assert "classical_bits" in df.columns


def test_trace_records_dataframe_amplitudes_without_classical_bits():
    dbg = CircuitDebugger(_simple_circuit())
    records = dbg.trace(include_initial=False)

    df = trace_records_to_dataframe(records, state_format="amplitudes", classical_bits=False)

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["step_index", "instruction", "amp_0", "amp_1"]
    # After applying H the amplitudes are 1/sqrt(2)
    amp0 = df.iloc[0]["amp_0"]
    amp1 = df.iloc[0]["amp_1"]
    assert amp0.real == pytest.approx(2**-0.5)
    assert amp0.imag == pytest.approx(0.0)
    assert amp1.real == pytest.approx(2**-0.5)
    assert amp1.imag == pytest.approx(0.0)


def test_trace_records_dataframe_classical_bit_columns():
    dbg = CircuitDebugger(_deterministic_circuit())
    records = dbg.trace(include_initial=True)

    df = trace_records_to_dataframe(records, classical_bit_columns=True)

    import pandas as pd

    assert list(df.columns) == [
        "step_index",
        "instruction",
        "classical_bits",
        "cbit_0",
        "p_0",
        "p_1",
    ]
    assert df["cbit_0"].dtype.name == "Int64"
    assert df["cbit_0"].isna().tolist()[:2] == [True, True]
    assert df.loc[df["instruction"] == "measure", "cbit_0"].iat[0] == 1


def test_probabilities_to_dataframe_columns():
    probs = [{"0": 1.0}, {"0": 0.25, "1": 0.75}]
    df = probabilities_to_dataframe(probs)

    assert list(df.columns) == ["step_index", "p_0", "p_1"]
    assert df.iloc[1]["p_1"] == pytest.approx(0.75)


def test_counts_to_dataframe_columns():
    counts = [{"0": 100}, {"0": 20, "1": 12}]
    df = counts_to_dataframe(counts)

    assert list(df.columns) == ["step_index", "c_0", "c_1"]
    assert df.iloc[1]["c_1"] == 12


def test_expectations_to_dataframe_roundtrip():
    rows = [{"X": 0.0, "Z": 1.0}, {"X": 0.5, "Z": 0.5}]
    df = expectations_to_dataframe(rows)

    assert list(df.columns) == ["step_index", "X", "Z"]
    assert df.iloc[1]["X"] == pytest.approx(0.5)
