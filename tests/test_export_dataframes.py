import numpy as np
import pytest
from qiskit import QuantumCircuit

pytest.importorskip("pandas")

from qiskit_inspect import (
    CircuitDebugger,
    TraceRecord,
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


def test_trace_records_dataframe_normalizes_probability_keys():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {0: 1.0},
            "classical_bits": [],
        },
        {
            "step_index": 1,
            "instruction": "measure",
            "state": {3: 0.75, 0: 0.25},
            "pre_measurement_state": {0: 0.4, 3: 0.6},
            "classical_bits": [],
        },
    ]

    df = trace_records_to_dataframe(dicts, state_format="probs", include_pre_measurement=True)

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert "p_00" in df.columns and "p_11" in df.columns
    assert "pre_p_00" in df.columns and "pre_p_11" in df.columns
    assert df.loc[0, "p_00"] == pytest.approx(1.0)
    assert df.loc[1, "p_11"] == pytest.approx(0.75)
    assert df.loc[1, "pre_p_00"] == pytest.approx(0.4)


def test_trace_records_dataframe_preserves_zero_bit_probabilities():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {"": 1.0},
            "classical_bits": [],
        },
        {
            "step_index": 1,
            "instruction": "measure",
            "state": {"0": 0.5, "1": 0.5},
            "classical_bits": [0],
        },
    ]

    df = trace_records_to_dataframe(dicts, state_format="probs")

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert "p_" in df.columns
    assert df.loc[0, "p_"] == pytest.approx(1.0)
    assert df.loc[1, "p_"] == pytest.approx(0.0)
    assert df.loc[1, "p_0"] == pytest.approx(0.5)


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


def test_trace_records_dataframe_probabilities_with_pre_measurement():
    dbg = CircuitDebugger(_simple_circuit())
    records = dbg.trace(include_initial=False)

    df = trace_records_to_dataframe(records, state_format="probs", include_pre_measurement=True)

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert "pre_p_0" in df.columns and "pre_p_1" in df.columns

    # The H gate row has no pre-measurement snapshot; pandas should report NA values.
    assert pd.isna(df.iloc[0]["pre_p_0"])
    assert pd.isna(df.iloc[0]["pre_p_1"])

    measure_row = df.iloc[-1]
    assert measure_row["pre_p_0"] == pytest.approx(0.5)
    assert measure_row["pre_p_1"] == pytest.approx(0.5)


def test_trace_records_dataframe_amplitudes_with_pre_measurement():
    dbg = CircuitDebugger(_simple_circuit())
    records = dbg.trace(include_initial=False)

    df = trace_records_to_dataframe(
        records,
        state_format="amplitudes",
        classical_bits=False,
        include_pre_measurement=True,
    )

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert "pre_amp_0" in df.columns and "pre_amp_1" in df.columns

    # Non-measurement rows should carry missing values in the pre-measurement columns.
    assert pd.isna(df.iloc[0]["pre_amp_0"])
    assert pd.isna(df.iloc[0]["pre_amp_1"])

    measure_row = df.iloc[-1]
    pre0 = measure_row["pre_amp_0"]
    pre1 = measure_row["pre_amp_1"]
    assert pre0.real == pytest.approx(2**-0.5)
    assert pre0.imag == pytest.approx(0.0)
    assert pre1.real == pytest.approx(2**-0.5)
    assert pre1.imag == pytest.approx(0.0)


def test_trace_records_dataframe_accepts_integer_like_probability_keys():
    from qiskit.quantum_info import Statevector

    class IntLike:
        def __init__(self, value: int) -> None:
            self.value = value

        def __index__(self) -> int:
            return self.value

        def __repr__(self) -> str:  # pragma: no cover - debugging aid
            return f"IntLike({self.value})"

    class IntKeyTraceRecord(TraceRecord):
        def to_dict(
            self,
            state_format: str = "probs",
            *,
            include_pre_measurement: bool = False,
        ) -> dict:
            base = super().to_dict(
                state_format=state_format, include_pre_measurement=include_pre_measurement
            )
            if state_format != "probs":
                return base
            base["state"] = {IntLike(0): 0.5, IntLike(3): 0.5}
            if include_pre_measurement:
                base["pre_measurement_state"] = {IntLike(0): 0.4, IntLike(1): 0.6}
            return base

    sv = Statevector.from_label("00")
    record = IntKeyTraceRecord(
        step_index=1,
        instruction="measure",
        state=sv,
        classical_bits=[0, 1],
        pre_measurement_state=sv.copy(),
    )

    df = trace_records_to_dataframe(
        [record],
        state_format="probs",
        include_pre_measurement=True,
    )

    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert {"p_00", "p_11"}.issubset(df.columns)
    assert {"pre_p_00", "pre_p_01"}.issubset(df.columns)
    assert df.loc[0, "p_00"] == pytest.approx(0.5)
    assert df.loc[0, "p_11"] == pytest.approx(0.5)
    assert df.loc[0, "pre_p_00"] == pytest.approx(0.4)
    assert df.loc[0, "pre_p_01"] == pytest.approx(0.6)


def test_trace_records_dataframe_rejects_probability_lists():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": [[1.0, 0.0]],
        }
    ]

    with pytest.raises(TypeError):
        trace_records_to_dataframe(dicts, state_format="probs")


def test_trace_records_dataframe_rejects_amplitude_mappings():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {"0": 1.0},
        }
    ]

    with pytest.raises(TypeError):
        trace_records_to_dataframe(dicts, state_format="amplitudes")


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


def test_trace_records_dataframe_accepts_numpy_bit_values():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [np.int64(1), np.bool_(False)],
            "state": {"0": 1.0},
        }
    ]

    df = trace_records_to_dataframe(dicts)

    assert df.iloc[0]["classical_bits"] == "10"


def test_trace_records_dataframe_accepts_numpy_array_bits():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": np.array([1, 0], dtype=int),
            "state": {"0": 1.0},
        }
    ]

    df = trace_records_to_dataframe(dicts)

    assert df.iloc[0]["classical_bits"] == "10"


def test_trace_records_dataframe_accepts_iterable_bit_values():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": (b for b in (None, 1, 0)),
            "state": {"0": 1.0},
        }
    ]

    df = trace_records_to_dataframe(dicts)

    assert df.iloc[0]["classical_bits"] == "x10"


def test_trace_records_dataframe_accepts_pandas_na_classical_bits():
    import pandas as pd

    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [pd.NA, 1, 0],
            "state": {"000": 1.0},
        }
    ]

    df = trace_records_to_dataframe(dicts)

    assert df.iloc[0]["classical_bits"] == "x10"


def test_trace_records_dataframe_accepts_unknown_string_entries():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": ["x", "1", "?", "0"],
            "state": {"0000": 1.0},
        }
    ]

    df = trace_records_to_dataframe(dicts)

    assert df.iloc[0]["classical_bits"] == "x1x0"


def test_trace_records_dataframe_requires_state_field():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [],
        }
    ]

    with pytest.raises(KeyError, match="required 'state' field"):
        trace_records_to_dataframe(dicts)


def test_trace_records_dataframe_rejects_none_state():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [],
            "state": None,
        }
    ]

    with pytest.raises(TypeError, match="'state' cannot be None"):
        trace_records_to_dataframe(dicts)


def test_trace_records_dataframe_rejects_non_binary_bit_values():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [0.5],
            "state": {"0": 1.0},
        }
    ]

    with pytest.raises(TypeError):
        trace_records_to_dataframe(dicts)


def test_trace_records_dataframe_rejects_non_binary_integer_values():
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [2, -1, np.int64(3)],
            "state": {"0": 1.0},
        }
    ]

    with pytest.raises(TypeError):
        trace_records_to_dataframe(dicts)


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
