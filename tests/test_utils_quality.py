import contextlib
import io
import json
import logging
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.result.utils import marginal_counts as qiskit_marginal_counts

from qiskit_inspect import CircuitDebugger
from qiskit_inspect.backend_trace import _extract_counts, _marginalize_counts
from qiskit_inspect.export import (
    write_expectations_csv,
    write_expectations_json,
    write_trace_csv,
    write_trace_json,
)
from qiskit_inspect.sampler_results import coerce_counts
from qiskit_inspect.trace_logging import enable_trace_logging
from qiskit_inspect.visual import ascii_histogram


@pytest.fixture(autouse=True)
def _reset_trace_logger():
    logger = logging.getLogger("qiskit_inspect.trace")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    try:
        yield
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()

        for handler in original_handlers:
            logger.addHandler(handler)

        logger.setLevel(original_level)
        logger.propagate = original_propagate


class _FakeJoinData:
    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class _FakePubResJoin:
    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def join_data(self):
        return _FakeJoinData(self._counts)


class _FakeMeas:
    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class _FakeData:
    def __init__(self, counts: Dict[str, int]):
        self.meas = _FakeMeas(counts)


class _FakePubResDataMeas:
    def __init__(self, counts: Dict[str, int]):
        self.data = _FakeData(counts)


class _FakePubResFallback:
    def __init__(self, counts: Dict[str, int]):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class _FakeUnsupported:
    pass


def _measurement_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def test_extract_counts_with_join_data():
    counts = {"0": 3, "1": 1}
    res = _FakePubResJoin(counts)
    out = _extract_counts(res)
    assert out == counts


def test_extract_counts_with_data_meas():
    counts = {"00": 2, "11": 2}
    res = _FakePubResDataMeas(counts)
    out = _extract_counts(res)
    assert out == counts


def test_extract_counts_fallback_direct():
    counts = {"010": 5}
    res = _FakePubResFallback(counts)
    out = _extract_counts(res)
    assert out == counts


def test_extract_counts_unsupported_raises():
    with pytest.raises(RuntimeError):
        _extract_counts(_FakeUnsupported())


def test_coerce_counts_accepts_integral_inputs():
    counts = {
        "0": 1,
        1: np.int64(2),
        "float": 3.0,
        "fraction": Fraction(4, 1),
        "bool": True,
        b"bytes": b"5",
        "decimal": Decimal("6"),
    }
    out = coerce_counts(counts)
    assert out == {
        "0": 1,
        "1": 2,
        "float": 3,
        "fraction": 4,
        "bool": 1,
        "b'bytes'": 5,
        "decimal": 6,
    }


def test_coerce_counts_accepts_iterable_of_pairs():
    items = [("0", 1), ("1", Fraction(2, 1))]
    out = coerce_counts(items)
    assert out == {"0": 1, "1": 2}


def test_coerce_counts_canonicalizes_sequence_keys():
    out = coerce_counts({(0, 1, 0): 3, (1, 0): 4})
    assert out["010"] == 3
    assert out["10"] == 4


def test_coerce_counts_normalizes_integer_keys_with_padding():
    out = coerce_counts({3: 7, 0: 9})
    assert out == {"11": 7, "00": 9}


def test_coerce_counts_merges_equivalent_bitstring_keys():
    out = coerce_counts({3: 2, "11": 5, (1, 1): 6})
    assert out == {"11": 13}


def test_coerce_counts_respects_explicit_zero_width_bitstrings():
    out = coerce_counts({"": 4, 0: 3})
    assert out == {"": 7}


def test_coerce_counts_rejects_fractional_inputs():
    with pytest.raises(TypeError):
        coerce_counts({"bad": 1.5})
    with pytest.raises(TypeError):
        coerce_counts({"bad": Fraction(3, 2)})
    with pytest.raises(TypeError):
        coerce_counts({"bad": Decimal("7.1")})
    with pytest.raises(TypeError):
        coerce_counts({"bad": Decimal("NaN")})


def test_coerce_counts_rejects_negative_values():
    with pytest.raises(ValueError):
        coerce_counts({"bad": -1})


def test_coerce_counts_rejects_non_mapping_iterables():
    with pytest.raises(TypeError):
        coerce_counts("01")


@pytest.mark.parametrize(
    ("counts", "keep_bits"),
    [
        ({"0100": 2, "1000": 3}, [2, 3]),
        ({"1 0": 4}, [0, 1]),
        ({"1_ 0 ": 7}, [0, 1]),
        ({"10": 5, "00": 2}, [0, 1]),
        ({"10": 5, "00": 2}, [1, 0]),
        ({"0": 1, "10": 2}, [0]),
        ({"0": 1}, [0, 0]),
        ({"0": 1.5}, [0]),
        ({"0": Fraction(3, 2)}, [0]),
        ({"0a": 1}, [0]),
    ],
)
def test_marginalize_counts_matches_qiskit(counts, keep_bits):
    expected = qiskit_marginal_counts(counts, keep_bits)
    assert _marginalize_counts(counts, keep_bits) == expected


@pytest.mark.parametrize(
    ("counts", "keep_bits"),
    [
        ({"0": 1}, [1]),
        ({"0": 1}, [0, 1]),
        ({"1": 2}, [2]),
    ],
)
def test_marginalize_counts_pads_before_matching_qiskit(counts, keep_bits):
    required_width = max(keep_bits) + 1
    padded: Dict[str, Any] = {}
    for raw_key, value in counts.items():
        cleaned = raw_key.replace(" ", "").replace("_", "")
        key = cleaned.rjust(required_width, "0")
        padded[key] = padded.get(key, 0) + value
    expected = qiskit_marginal_counts(padded, keep_bits)
    assert _marginalize_counts(counts, keep_bits) == expected


@pytest.mark.parametrize(
    ("counts", "keep_bits"),
    [
        ({"0": 1}, []),
        ({0: 1}, [0]),
        ({("1", "0"): 2}, [0]),
        ({}, [0]),
        ("01", [0]),
        ({"0": 1}, 1),
    ],
)
def test_marginalize_counts_exceptions_match_qiskit(counts, keep_bits):
    with pytest.raises(Exception) as expected:
        qiskit_marginal_counts(counts, keep_bits)
    with pytest.raises(type(expected.value)) as actual:
        _marginalize_counts(counts, keep_bits)
    assert str(actual.value) == str(expected.value)


def test_ascii_histogram_empty():
    assert ascii_histogram({}) == ""


def test_ascii_histogram_normalizes_and_sorts():
    txt = ascii_histogram({"10": 2.0, "00": 1.0}, width=10, sort=True)
    assert txt == "10: ########## 0.6667\n00: ##### 0.3333"


def test_ascii_histogram_respects_input_order_without_sort():
    txt = ascii_histogram({"a": 0.1, "b": 0.9}, width=4, sort=False)
    assert txt.splitlines()[0].startswith("a:")


def test_ascii_histogram_handles_zero_width():
    assert ascii_histogram({"0": 1.0}, width=0) == "0:  1.0000"


def test_write_expectations_csv_and_json(tmp_path):
    rows = [{"A": 0.1, "B": 0.2}, {"A": 0.3}]
    csv_file = tmp_path / "exp.csv"
    json_file = tmp_path / "exp.json"

    write_expectations_csv(rows, str(csv_file))
    write_expectations_json(rows, str(json_file))

    assert csv_file.exists() and csv_file.read_text(encoding="utf-8").startswith("step_index")
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data == rows


def test_write_trace_csv_and_json_from_dicts(tmp_path):
    # Minimal dicts as produced by TraceRecord.to_dict(state_format="probs")
    dicts = [
        {"step_index": 0, "instruction": None, "classical_bits": [None, 1], "state": {"0": 1.0}},
        {
            "step_index": 1,
            "instruction": "h",
            "classical_bits": [0, 1],
            "state": {"00": 0.5, "11": 0.5},
        },
    ]
    csv_file = tmp_path / "trace.csv"
    json_file = tmp_path / "trace.json"

    write_trace_csv(dicts, str(csv_file), state_format="probs")
    write_trace_json(dicts, str(json_file), state_format="probs")

    # CSV should have p_<bitstring> columns and formatted classical bits string
    import csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header[3:] == ["p_00", "p_11"]

    # JSON normalizes probability keys to the widest basis width
    out = json.loads(json_file.read_text(encoding="utf-8"))
    assert out[0]["state"] == {"00": pytest.approx(1.0)}
    assert set(out[1]["state"].keys()) == {"00", "11"}
    assert out[1]["state"]["11"] == pytest.approx(0.5)


def test_write_trace_csv_rejects_non_probability_state(tmp_path):
    dicts = [
        {
            "step_index": 0,
            "instruction": "measure",
            "classical_bits": [0],
            "state": [[1.0, 0.0]],
        }
    ]

    with pytest.raises(TypeError, match="state data"):
        write_trace_csv(dicts, str(tmp_path / "trace.csv"))


def test_write_trace_csv_accepts_string_classical_bits(tmp_path):
    dicts = [
        {
            "step_index": 0,
            "instruction": None,
            "classical_bits": "x10",
            "state": {"000": 1.0},
        },
        {
            "step_index": 1,
            "instruction": "measure",
            "classical_bits": "101",
            "state": {"101": 0.75, "001": 0.25},
        },
    ]

    csv_file = tmp_path / "trace_bits.csv"
    write_trace_csv(dicts, str(csv_file))

    import csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[1][2] == "x10"
    assert rows[2][2] == "101"


def test_write_trace_csv_accepts_sequence_unknown_bits(tmp_path):
    dicts = [
        {
            "step_index": 0,
            "instruction": None,
            "classical_bits": ["x", "1", "?", "0"],
            "state": {"0000": 1.0},
        }
    ]

    csv_file = tmp_path / "trace_seq_bits.csv"
    write_trace_csv(dicts, str(csv_file))

    import csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[1][2] == "x1x0"


def test_write_trace_csv_normalizes_probability_keys(tmp_path):
    dicts = [
        {
            "step_index": 1,
            "instruction": "measure",
            "classical_bits": [1],
            "state": {0: 0.25, 3: 0.75},
            "pre_measurement_state": {3: 0.6, 0: 0.4},
        }
    ]

    csv_file = tmp_path / "trace_int_keys.csv"
    write_trace_csv(dicts, str(csv_file), include_pre_measurement=True)

    import csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    assert "p_00" in header and "p_11" in header
    assert "pre_p_00" in header and "pre_p_11" in header

    values = rows[1]
    p00_index = header.index("p_00")
    p11_index = header.index("p_11")
    pre00_index = header.index("pre_p_00")
    pre11_index = header.index("pre_p_11")

    assert float(values[p00_index]) == pytest.approx(0.25)
    assert float(values[p11_index]) == pytest.approx(0.75)
    assert float(values[pre00_index]) == pytest.approx(0.4)
    assert float(values[pre11_index]) == pytest.approx(0.6)


def test_write_trace_json_from_records_includes_pre_measurement(tmp_path):
    dbg = CircuitDebugger(_measurement_circuit())
    records = dbg.trace(include_initial=False)

    json_file = tmp_path / "trace_pre.json"
    write_trace_json(records, str(json_file), include_pre_measurement=True)

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "pre_measurement_state" not in data[0]
    pre = data[-1]["pre_measurement_state"]
    assert pre["0"] == pytest.approx(0.5)
    assert pre["1"] == pytest.approx(0.5)


def test_write_trace_json_amplitudes_include_pre_measurement(tmp_path):
    dbg = CircuitDebugger(_measurement_circuit())
    records = dbg.trace(include_initial=False)

    json_file = tmp_path / "trace_pre_amp.json"
    write_trace_json(
        records,
        str(json_file),
        state_format="amplitudes",
        include_pre_measurement=True,
    )

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "pre_measurement_state" not in data[0]
    pre = data[-1]["pre_measurement_state"]
    assert len(pre) == 2
    expected = 2**-0.5
    for real, imag in pre:
        assert float(real) == pytest.approx(expected)
        assert float(imag) == pytest.approx(0.0)


def test_write_trace_json_strips_pre_measurement_when_disabled(tmp_path):
    dbg = CircuitDebugger(_measurement_circuit())
    records = dbg.trace(include_initial=False)

    dicts = [rec.to_dict(include_pre_measurement=True) for rec in records]

    json_file = tmp_path / "trace_no_pre.json"
    write_trace_json(dicts, str(json_file), include_pre_measurement=False)

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert all("pre_measurement_state" not in entry for entry in data)


def test_write_trace_json_amplitudes_strip_pre_measurement_when_disabled(tmp_path):
    dbg = CircuitDebugger(_measurement_circuit())
    records = dbg.trace(include_initial=False)

    dicts = [
        rec.to_dict(state_format="amplitudes", include_pre_measurement=True) for rec in records
    ]

    json_file = tmp_path / "trace_no_pre_amp.json"
    write_trace_json(
        dicts,
        str(json_file),
        state_format="amplitudes",
        include_pre_measurement=False,
    )

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert all("pre_measurement_state" not in entry for entry in data)


def test_write_trace_json_normalizes_probability_keys(tmp_path):
    dicts = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {0: 1.0},
        },
        {
            "step_index": 1,
            "instruction": "measure",
            "state": {3: 0.75, 0: 0.25},
            "pre_measurement_state": {0: 0.4, 3: 0.6},
        },
    ]

    json_file = tmp_path / "trace_int_keys.json"
    write_trace_json(dicts, str(json_file), include_pre_measurement=True)

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert set(data[0]["state"].keys()) == {"00"}
    assert set(data[1]["state"].keys()) == {"00", "11"}
    pre_keys = set(data[1]["pre_measurement_state"].keys())
    assert pre_keys == {"00", "11"}
    assert data[1]["state"]["11"] == pytest.approx(0.75)
    assert data[1]["pre_measurement_state"]["00"] == pytest.approx(0.4)


def test_write_trace_json_requires_state_field(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [],
        }
    ]

    with pytest.raises(KeyError, match="required 'state' field"):
        write_trace_json(records, str(tmp_path / "missing_state.json"))


def test_write_trace_json_rejects_none_state(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "classical_bits": [],
            "state": None,
        }
    ]

    with pytest.raises(TypeError, match="'state' cannot be None"):
        write_trace_json(records, str(tmp_path / "none_state.json"))


def test_write_trace_json_rejects_probability_lists(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": [[1.0, 0.0]],
        }
    ]

    with pytest.raises(TypeError, match="probability exports require mappings"):
        write_trace_json(records, str(tmp_path / "trace_bad.json"), state_format="probs")


def test_write_trace_json_rejects_non_numeric_probabilities(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {"0": "1.0"},
        }
    ]

    with pytest.raises(TypeError, match="string probability value"):
        write_trace_json(records, str(tmp_path / "trace_bad_numeric.json"), state_format="probs")


def test_write_trace_json_rejects_non_finite_probabilities(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {"0": float("nan")},
        }
    ]

    with pytest.raises(TypeError, match="non-finite probability"):
        write_trace_json(records, str(tmp_path / "trace_bad_finite.json"), state_format="probs")


def test_write_trace_json_rejects_amplitude_mappings(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": {"0": 1.0},
        }
    ]

    with pytest.raises(TypeError, match="amplitude exports require sequences"):
        write_trace_json(records, str(tmp_path / "trace_bad_amp.json"), state_format="amplitudes")


def test_write_trace_json_rejects_non_finite_amplitudes(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "noop",
            "state": [[float("inf"), 0.0]],
        }
    ]

    with pytest.raises(TypeError, match="non-finite amplitude"):
        write_trace_json(
            records, str(tmp_path / "trace_bad_amp_val.json"), state_format="amplitudes"
        )


def test_write_trace_json_rejects_pre_measurement_mismatch(tmp_path):
    records = [
        {
            "step_index": 0,
            "instruction": "measure",
            "state": {"0": 1.0},
            "pre_measurement_state": {"0": 1.0},
        }
    ]

    with pytest.raises(TypeError, match="amplitude exports require sequences"):
        write_trace_json(
            records,
            str(tmp_path / "trace_bad_pre.json"),
            state_format="amplitudes",
            include_pre_measurement=True,
        )


def test_write_trace_csv_from_records_includes_pre_measurement(tmp_path):
    dbg = CircuitDebugger(_measurement_circuit())
    records = dbg.trace(include_initial=False)

    csv_file = tmp_path / "trace_pre.csv"
    write_trace_csv(records, str(csv_file), include_pre_measurement=True)

    import csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    assert "pre_p_0" in header and "pre_p_1" in header

    pre0_idx = header.index("pre_p_0")
    pre1_idx = header.index("pre_p_1")

    h_row = rows[1]
    # Non-measurement rows should leave pre-measurement columns blank.
    assert h_row[pre0_idx] == ""
    assert h_row[pre1_idx] == ""

    measure_row = rows[-1]
    assert float(measure_row[pre0_idx]) == pytest.approx(0.5)
    assert float(measure_row[pre1_idx]) == pytest.approx(0.5)


def test_enable_trace_logging_idempotent(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    logger = enable_trace_logging(level=logging.INFO, propagate=False)
    # second call should not add extra handlers; should reuse/update levels
    logger2 = enable_trace_logging(level=logging.DEBUG, propagate=False)
    assert logger is logger2
    # ensure there's exactly one StreamHandler attached
    handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(handlers) == 1
    # handler level should match the latest level
    assert handlers[0].level == logging.DEBUG


def test_enable_trace_logging_updates_stream():
    stream1 = io.StringIO()
    stream2 = io.StringIO()

    logger = enable_trace_logging(level=logging.INFO, stream=stream1, propagate=False)
    logger.info("first")
    assert "first" in stream1.getvalue()

    # Second call with a new stream should retarget existing handlers.
    enable_trace_logging(level=logging.INFO, stream=stream2, propagate=False)
    logger.info("second")

    assert "second" not in stream1.getvalue()
    assert "second" in stream2.getvalue()


def test_enable_trace_logging_adds_stream_handler_if_missing():
    logger = logging.getLogger("qiskit_inspect.trace")
    logger.addHandler(logging.NullHandler())

    enable_trace_logging(level=logging.INFO, propagate=False)

    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers
    assert stream_handlers[0].level == logging.INFO


def test_enable_trace_logging_defaults_to_current_stderr(monkeypatch):
    import sys

    stderr1 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr1)

    logger = enable_trace_logging(level=logging.INFO, propagate=False)
    logger.info("first")
    assert "first" in stderr1.getvalue()

    stream = io.StringIO()
    enable_trace_logging(level=logging.INFO, stream=stream, propagate=False)
    logger.info("second")
    assert "second" in stream.getvalue()

    stderr2 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr2)
    enable_trace_logging(level=logging.INFO, propagate=False)
    logger.info("third")

    assert "third" not in stream.getvalue()
    assert "third" in stderr2.getvalue()
