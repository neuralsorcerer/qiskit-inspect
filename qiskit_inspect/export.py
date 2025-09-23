"""Export utilities for expectations and execution traces."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable as IterableABC, Mapping as MappingABC, Sequence as SequenceABC
from numbers import Integral, Real
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from .debugger import TraceRecord
from .probabilities import canonicalize_bitstring_key, normalize_probability_dict
from .visual import format_classical_bits

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import pandas as pd  # type: ignore[import-untyped]


def write_expectations_csv(rows: List[Dict[str, float]], file_path: str) -> None:
    """Write expectation rows to CSV.

    Args:
        rows: Sequence of mappings ``observable_name -> value`` (one per prefix).
            Missing names default to ``0.0``.
        file_path: Destination CSV path.

    Notes:
        The CSV columns are: ``step_index`` followed by the sorted observable names.
    """
    # Determine union of keys
    keys: Set[str] = set()
    for r in rows:
        keys.update(r.keys())
    header = ["step_index"] + sorted(keys)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([i] + [r.get(k, 0.0) for k in header[1:]])


def write_expectations_json(rows: List[Dict[str, float]], file_path: str) -> None:
    """Write expectation rows to a JSON file (pretty-printed).

    Args:
        rows: Sequence of mappings ``observable_name -> value``.
        file_path: Destination JSON path.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _validate_serialized_state_field(
    record: Mapping[str, object],
    *,
    field: str,
    state_format: str,
    index: int,
) -> None:
    """Validate the serialized state stored under ``field`` for ``state_format``."""

    if state_format not in {"probs", "amplitudes"}:
        raise ValueError("state_format must be 'probs' or 'amplitudes'")

    value = record.get(field)
    if value is None:
        return

    if state_format == "probs":
        if value == {}:
            return
        if not isinstance(value, MappingABC):
            raise TypeError(
                "Trace record at index "
                f"{index} has {field} data of type {type(value).__name__}; "
                "probability exports require mappings."
            )
        for key, raw in value.items():
            if isinstance(raw, (str, bytes)):
                raise TypeError(
                    "Trace record at index "
                    f"{index} has a string probability value for key {key!r} under {field}; "
                    "probabilities must be numeric."
                )
            try:
                numeric = float(raw)
            except Exception as exc:
                raise TypeError(
                    "Trace record at index "
                    f"{index} has a non-numeric probability value for key {key!r} under {field}."
                ) from exc
            if not math.isfinite(numeric):
                raise TypeError(
                    "Trace record at index "
                    f"{index} contains a non-finite probability value for key {key!r} under {field}."
                )
        return

    # state_format == "amplitudes"
    if value == []:
        return
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise TypeError(
            "Trace record at index "
            f"{index} has {field} data of type {type(value).__name__}; "
            "amplitude exports require sequences of [re, im] pairs."
        )
    for entry in value:
        if not isinstance(entry, SequenceABC) or isinstance(entry, (str, bytes)) or len(entry) != 2:
            raise TypeError(
                "Trace record at index "
                f"{index} has malformed amplitude data under {field}; "
                "entries must be length-2 sequences of real numbers."
            )
        try:
            real_part = float(entry[0])
            imag_part = float(entry[1])
        except Exception as exc:
            raise TypeError(
                "Trace record at index "
                f"{index} contains non-numeric amplitude components under {field}."
            ) from exc
        if not math.isfinite(real_part) or not math.isfinite(imag_part):
            raise TypeError(
                "Trace record at index "
                f"{index} contains non-finite amplitude components under {field}."
            )


def _ensure_trace_dicts(
    records: Iterable[Union[TraceRecord, Mapping[str, object]]],
    state_format: str,
    *,
    include_pre_measurement: bool = False,
) -> List[dict]:
    """Convert a sequence of ``TraceRecord``/dict into a list of dicts.

    Args:
        records: Iterable of :class:`TraceRecord` or already-serialized dicts.
        state_format: Passed to :meth:`TraceRecord.to_dict` when serializing.

    Returns:
        list[dict]: Plain dicts representing trace records.
    """

    if state_format not in {"probs", "amplitudes"}:
        raise ValueError("state_format must be 'probs' or 'amplitudes'")

    out: List[dict] = []
    probability_dicts: List[Mapping[str, float]] = []
    pre_probability_dicts: List[Mapping[str, float]] = []
    for index, r in enumerate(records):
        if isinstance(r, TraceRecord):
            serialized = r.to_dict(
                state_format=state_format,
                include_pre_measurement=include_pre_measurement,
            )
            if "state" not in serialized:
                raise KeyError(
                    f"Trace record at index {index} is missing the required 'state' field."
                )
            if serialized.get("state") is None:
                raise TypeError(
                    f"Trace record at index {index} produced a None 'state' value; state data is required."
                )
            if state_format == "probs":
                state_value = serialized.get("state")
                if isinstance(state_value, MappingABC):
                    probability_dicts.append(state_value)
                pre_value = serialized.get("pre_measurement_state")
                if isinstance(pre_value, MappingABC):
                    pre_probability_dicts.append(pre_value)
            out.append(serialized)
        else:
            if not isinstance(r, MappingABC):
                raise TypeError(
                    "Trace records must be TraceRecord instances or mappings; "
                    f"encountered {type(r).__name__} at index {index}."
                )
            mapping = dict(r)
            if "state" not in mapping:
                raise KeyError(
                    f"Trace record at index {index} is missing the required 'state' field."
                )
            if mapping.get("state") is None:
                raise TypeError(
                    f"Trace record at index {index} must include state data; 'state' cannot be None."
                )
            if not include_pre_measurement:
                mapping.pop("pre_measurement_state", None)
            _validate_serialized_state_field(
                mapping,
                field="state",
                state_format=state_format,
                index=index,
            )
            if include_pre_measurement or "pre_measurement_state" in mapping:
                _validate_serialized_state_field(
                    mapping,
                    field="pre_measurement_state",
                    state_format=state_format,
                    index=index,
                )
            if state_format == "probs":
                state_value = mapping.get("state")
                if isinstance(state_value, MappingABC):
                    normalized_state = normalize_probability_dict(state_value)
                    mapping["state"] = normalized_state
                    probability_dicts.append(normalized_state)
                pre_value = mapping.get("pre_measurement_state")
                if isinstance(pre_value, MappingABC):
                    normalized_pre = normalize_probability_dict(pre_value)
                    mapping["pre_measurement_state"] = normalized_pre
                    pre_probability_dicts.append(normalized_pre)
            out.append(mapping)
    if state_format == "probs" and out:

        def _has_non_empty_bitstrings(mapping: Mapping[str, object]) -> bool:
            for key in mapping.keys():
                if isinstance(key, str):
                    if key:
                        return True
                else:
                    if str(key):
                        return True
            return False

        def _bitstring_width(key: object) -> int:
            try:
                label = canonicalize_bitstring_key(key)
            except Exception:
                label = str(key)
            return len(label)

        max_width = 0
        for mapping in probability_dicts:
            if mapping:
                max_width = max(max_width, max(_bitstring_width(k) for k in mapping.keys()))
        for mapping in pre_probability_dicts:
            if mapping:
                max_width = max(max_width, max(_bitstring_width(k) for k in mapping.keys()))
        if max_width:
            for mapping in out:
                state_value = mapping.get("state")
                if isinstance(state_value, MappingABC) and _has_non_empty_bitstrings(state_value):
                    mapping["state"] = normalize_probability_dict(state_value, num_qubits=max_width)
                pre_value = mapping.get("pre_measurement_state")
                if isinstance(pre_value, MappingABC) and _has_non_empty_bitstrings(pre_value):
                    mapping["pre_measurement_state"] = normalize_probability_dict(
                        pre_value, num_qubits=max_width
                    )
    return out


def _require_pandas():
    """Import :mod:`pandas` on demand and raise a friendly error if missing."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in integration tests
        raise RuntimeError(
            "pandas is required for DataFrame exports. Install it with "
            "'pip install qiskit-inspect[data]' or 'pip install pandas'."
        ) from exc
    return pd


def _is_pandas_na(value: object) -> bool:
    """Return ``True`` when ``value`` is pandas' NA sentinel."""

    if value is None:
        return False
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    if not module.startswith("pandas"):
        return False
    return value_type.__name__ == "NAType"


def _coerce_classical_bits(value: object) -> List[Optional[int]]:
    """Return ``value`` as a list of optional integers describing classical bits."""

    if value is None:
        return []
    if _is_pandas_na(value):
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        out: List[Optional[int]] = []
        for ch in cleaned:
            if ch in {"x", "X", "?"}:
                out.append(None)
            elif ch in {"0", "1"}:
                out.append(int(ch))
            elif ch.isspace() or ch == "_":
                continue
            else:
                raise ValueError(
                    "Classical bit strings must contain only 0, 1, x, X, or ? characters."
                )
        return out

    sequence: Sequence[object] | None = None
    if isinstance(value, (list, tuple)):
        sequence = value
    elif isinstance(value, MappingABC):
        raise TypeError("Classical bit values must be provided as a sequence or string.")
    else:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                maybe_sequence = tolist()
            except Exception:
                maybe_sequence = None
            else:
                if isinstance(maybe_sequence, list):
                    sequence = maybe_sequence
                elif maybe_sequence is not None:
                    sequence = [maybe_sequence]
        if (
            sequence is None
            and isinstance(value, IterableABC)
            and not isinstance(value, (bytes, str))
        ):
            sequence = list(value)
    if sequence is None:
        raise TypeError("Classical bit values must be provided as a sequence or string.")

    out: List[Optional[int]] = []
    for item in sequence:
        if item is None:
            out.append(None)
            continue
        if _is_pandas_na(item):
            out.append(None)
            continue

        text_value: str | None = None
        if isinstance(item, str):
            text_value = item.strip()
        elif isinstance(item, bytes):
            try:
                text_value = item.decode("utf-8").strip()
            except Exception as exc:
                raise TypeError(
                    "Classical bit entries provided as bytes must be valid UTF-8 strings."
                ) from exc

        if text_value is not None:
            if text_value in {"x", "X", "?"}:
                out.append(None)
                continue
            if text_value in {"0", "1"}:
                out.append(int(text_value))
                continue

        if isinstance(item, Integral):
            coerced = int(item)
            if coerced not in (0, 1):
                raise TypeError(
                    "Classical bit entries must be 0, 1, None, or strings containing 0/1."
                )
            out.append(coerced)
            continue

        dtype = getattr(item, "dtype", None)
        if dtype is not None and str(dtype) == "bool":
            out.append(int(bool(item)))
            continue
        if isinstance(item, Real):
            numeric = float(item)
            if numeric in (0.0, 1.0):
                out.append(int(numeric))
                continue
        raise TypeError("Classical bit entries must be 0, 1, None, or strings containing 0/1.")
    return out


def write_trace_csv(
    records: Iterable[Union[TraceRecord, dict]],
    file_path: str,
    state_format: str = "probs",
    *,
    include_pre_measurement: bool = False,
) -> None:
    """Write trace snapshots to CSV.

    Args:
        records: Iterable of :class:`TraceRecord` or dicts.
        file_path: Destination CSV path.
        state_format: Only ``"probs"`` is supported for CSV (amplitudes are not
            supported in this format).
        include_pre_measurement: When ``True``, emit additional columns for
            pre-measurement probabilities when present.

    Raises:
        ValueError: If ``state_format`` is not ``"probs"``.

    Notes:
        Columns: ``step_index``, ``instruction``, ``classical_bits`` (string),
        followed by probability columns named ``p_<bitstring>``.  When
        ``include_pre_measurement`` is ``True`` and a record contains
        ``pre_measurement_state`` data, additional ``pre_p_<bitstring>`` columns
        are added containing the pre-collapse probabilities.
    """
    if state_format != "probs":
        raise ValueError("write_trace_csv supports only state_format='probs'")
    dicts = _ensure_trace_dicts(
        records,
        state_format=state_format,
        include_pre_measurement=include_pre_measurement,
    )

    def _require_probability_mapping(record: dict, field: str, index: int) -> Mapping[str, object]:
        raw = record.get(field)
        if raw is None or raw == {}:
            return {}
        if not isinstance(raw, MappingABC):
            raise TypeError(
                "Trace record at index "
                f"{index} has {field} data of type {type(raw).__name__}; "
                "write_trace_csv requires probability dictionaries. "
                "Re-serialize the records with state_format='probs'."
            )
        return raw

    # Union of probability keys
    pkeys: Set[str] = set()
    for idx, d in enumerate(dicts):
        p = _require_probability_mapping(d, "state", idx)
        pkeys.update(p.keys())
    pcols = [f"p_{k}" for k in sorted(pkeys)]
    pre_cols: List[str] = []
    if include_pre_measurement:
        pre_keys: Set[str] = set()
        for idx, d in enumerate(dicts):
            if "pre_measurement_state" not in d or d.get("pre_measurement_state") is None:
                continue
            pre_state = _require_probability_mapping(d, "pre_measurement_state", idx)
            pre_keys.update(pre_state.keys())
        pre_cols = [f"pre_p_{k}" for k in sorted(pre_keys)]

    header = ["step_index", "instruction", "classical_bits"] + pcols + pre_cols
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx, d in enumerate(dicts):
            step = d.get("step_index")
            instr = d.get("instruction")
            cbits = _coerce_classical_bits(d.get("classical_bits", []))
            cbits_str = format_classical_bits(cbits)
            p = _require_probability_mapping(d, "state", idx)
            row = [step, instr, cbits_str] + [p.get(k[2:], 0.0) for k in pcols]
            if pre_cols:
                if "pre_measurement_state" not in d or d.get("pre_measurement_state") is None:
                    row.extend("" for _ in pre_cols)
                else:
                    pre_state = _require_probability_mapping(d, "pre_measurement_state", idx)
                    row.extend(pre_state.get(name[6:], 0.0) for name in pre_cols)
            w.writerow(row)


def write_trace_json(
    records: Iterable[Union[TraceRecord, dict]],
    file_path: str,
    state_format: str = "probs",
    *,
    include_pre_measurement: bool = False,
) -> None:
    """Write trace snapshots to a JSON file.

    If ``records`` contains :class:`TraceRecord` objects, they are serialized using
    the requested ``state_format`` before writing.

    Args:
        records: Iterable of :class:`TraceRecord` or dicts.
        file_path: Destination JSON path.
        state_format: Serialization format for the state (``"probs"`` or
            ``"amplitudes"``).
        include_pre_measurement: When ``True``, serialize the
            ``pre_measurement_state`` field for measurement records using the
            requested ``state_format``.
    """
    dicts = _ensure_trace_dicts(
        records,
        state_format=state_format,
        include_pre_measurement=include_pre_measurement,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2)


def trace_records_to_dataframe(
    records: Iterable[Union[TraceRecord, dict]],
    *,
    state_format: str = "probs",
    classical_bits: bool = True,
    classical_bit_columns: bool | Sequence[str] = False,
    include_pre_measurement: bool = False,
) -> "pd.DataFrame":
    """Return a :class:`pandas.DataFrame` summarizing debugger trace records.

    Args:
        records: Sequence of :class:`TraceRecord` objects or dictionaries as
            returned by :meth:`TraceRecord.to_dict`.
        state_format: ``"probs"`` to include probability columns (``p_<bit>``)
            or ``"amplitudes"`` to include complex amplitudes (``amp_<index>``).
        classical_bits: If ``True`` (default), include a ``classical_bits``
            column with the formatted classical register snapshot for each
            record.  Disable to omit the column when it is not required.
        classical_bit_columns: If ``True``, add one column per classical bit
            using ``cbit_<index>`` names (starting at 0).  Provide an explicit
            sequence of strings to control the column labels.  Unknown bit
            values are represented as ``pd.NA`` in these columns.
        include_pre_measurement: When ``True``, include columns for
            ``pre_measurement_state`` snapshots.  For probability exports the
            columns are named ``pre_p_<bitstring>``; for amplitude exports the
            columns follow ``pre_amp_<index>``.

    Returns:
        pandas.DataFrame: A data frame with one row per trace record ordered as
        provided.
    """

    pd = _require_pandas()

    dicts = _ensure_trace_dicts(
        records,
        state_format=state_format,
        include_pre_measurement=include_pre_measurement,
    )
    base_columns: List[str] = ["step_index", "instruction"]

    def _flatten_probabilities(
        field: str, prefix: str
    ) -> tuple[List[str], List[Dict[str, float]], List[bool]]:
        all_keys: Set[str] = set()
        flattened_list: List[Dict[str, float]] = []
        present_flags: List[bool] = []
        for index, d in enumerate(dicts):
            present = field in d and d[field] is not None
            present_flags.append(present)
            raw = d.get(field)
            if raw is None or raw == {}:
                entry: Dict[str, float] = {}
            elif not isinstance(raw, Mapping):
                raise TypeError(
                    "Trace record at index "
                    f"{index} has {field} data of type {type(raw).__name__}; "
                    "trace_records_to_dataframe requires probability dictionaries when "
                    "state_format='probs'."
                )
            else:
                entry = {f"{prefix}{key}": float(value) for key, value in raw.items()}
            flattened_list.append(entry)
            all_keys.update(entry.keys())
        return sorted(all_keys), flattened_list, present_flags

    def _flatten_amplitudes(
        field: str, prefix: str
    ) -> tuple[List[str], List[Dict[str, complex]], List[bool]]:
        max_len = 0
        flattened_list: List[Dict[str, complex]] = []
        present_flags: List[bool] = []
        for index, d in enumerate(dicts):
            present = field in d and d[field] is not None
            present_flags.append(present)
            raw = d.get(field)
            if raw is None:
                raw_seq: Sequence[object] = []
            elif not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise TypeError(
                    "Trace record at index "
                    f"{index} has {field} data of type {type(raw).__name__}; "
                    "trace_records_to_dataframe requires amplitude sequences when "
                    "state_format='amplitudes'."
                )
            else:
                raw_seq = raw
            amp_entry: Dict[str, complex] = {}
            for idx, pair in enumerate(raw_seq):
                real_imag = pair
                if not isinstance(real_imag, Sequence) or len(real_imag) != 2:
                    raise ValueError("Amplitude entries must be [re, im] pairs.")
                amp_entry[f"{prefix}{idx}"] = complex(float(real_imag[0]), float(real_imag[1]))
            flattened_list.append(amp_entry)
            max_len = max(max_len, len(raw_seq))
        columns = [f"{prefix}{i}" for i in range(max_len)]
        return columns, flattened_list, present_flags

    if state_format == "probs":
        state_columns, flattened_states, _ = _flatten_probabilities("state", "p_")
        if include_pre_measurement:
            pre_columns, flattened_pre_states, pre_presence = _flatten_probabilities(
                "pre_measurement_state", "pre_p_"
            )
        else:
            pre_columns, flattened_pre_states, pre_presence = (
                [],
                [{} for _ in dicts],
                [False] * len(dicts),
            )
    elif state_format == "amplitudes":
        state_columns, flattened_states, _ = _flatten_amplitudes("state", "amp_")
        if include_pre_measurement:
            pre_columns, flattened_pre_states, pre_presence = _flatten_amplitudes(
                "pre_measurement_state", "pre_amp_"
            )
        else:
            pre_columns, flattened_pre_states, pre_presence = (
                [],
                [{} for _ in dicts],
                [False] * len(dicts),
            )
    else:
        raise ValueError("state_format must be 'probs' or 'amplitudes'")

    classical_lists: List[List[Optional[int]]] = []
    max_bits = 0
    for d in dicts:
        cbits = _coerce_classical_bits(d.get("classical_bits", []))
        classical_lists.append(cbits)
        max_bits = max(max_bits, len(cbits))

    bit_column_names: List[str] = []
    if classical_bit_columns:
        if isinstance(classical_bit_columns, bool):
            bit_column_names = [f"cbit_{i}" for i in range(max_bits)]
        else:
            bit_column_names = [str(name) for name in classical_bit_columns]
            if len(bit_column_names) != max_bits:
                raise ValueError(
                    "classical_bit_columns must provide exactly one label per classical bit."
                )
        if len(bit_column_names) != max_bits:
            raise ValueError(
                "classical_bit_columns requested but the provided labels do not match the number "
                "of classical bits present in the records."
            )

    if classical_bits:
        base_columns.append("classical_bits")
    columns = base_columns + bit_column_names + state_columns + pre_columns

    rows: List[List[Optional[Union[int, str, float, complex]]]] = []
    for d, cbits, state_values, pre_values, has_pre in zip(
        dicts, classical_lists, flattened_states, flattened_pre_states, pre_presence
    ):
        row: List[Optional[Union[int, str, float, complex]]] = [
            d.get("step_index"),
            d.get("instruction"),
        ]
        if classical_bits:
            row.append(format_classical_bits(cbits))
        if bit_column_names:
            padded = list(cbits) + [None] * (len(bit_column_names) - len(cbits))
            row.extend(padded)
        for col in state_columns:
            row.append(state_values.get(col))
        for col in pre_columns:
            if has_pre:
                default_value: Union[float, complex]
                if state_format == "probs":
                    default_value = 0.0
                else:
                    default_value = 0j
                row.append(pre_values.get(col, default_value))
            else:
                row.append(None)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    if bit_column_names:
        for name in bit_column_names:
            df[name] = pd.array(df[name], dtype="Int64")

    if state_columns:
        fill_value: Union[float, complex]
        if state_format == "probs":
            fill_value = 0.0
        else:
            fill_value = 0j
        df.loc[:, state_columns] = df[state_columns].fillna(fill_value)

    return df


def _sequence_of_dicts_to_dataframe(
    rows: Sequence[Mapping[str, Union[int, float]]],
    *,
    index_label: str,
) -> "pd.DataFrame":
    pd = _require_pandas()
    keys: Set[str] = set()
    for r in rows:
        keys.update(r.keys())
    columns = [index_label] + sorted(keys)
    data: List[List[Union[int, float]]] = []
    for i, row in enumerate(rows):
        data.append([i] + [row.get(k, 0) for k in columns[1:]])
    return pd.DataFrame(data, columns=columns)


def expectations_to_dataframe(rows: Sequence[Dict[str, float]]) -> "pd.DataFrame":
    """Return expectation values as a :class:`pandas.DataFrame`.

    The resulting frame includes a ``step_index`` column that matches the order
    of the input sequence and one column per observable name.
    """

    return _sequence_of_dicts_to_dataframe(rows, index_label="step_index")


def probabilities_to_dataframe(rows: Sequence[Dict[str, float]]) -> "pd.DataFrame":
    """Return probability dictionaries as a :class:`pandas.DataFrame`.

    Each bitstring key becomes a column named ``p_<bitstring>`` and a
    ``step_index`` column identifies the prefix order.
    """

    prefixed = [{f"p_{k}": float(v) for k, v in row.items()} for row in rows]
    return _sequence_of_dicts_to_dataframe(prefixed, index_label="step_index")


def counts_to_dataframe(rows: Sequence[Dict[str, int]]) -> "pd.DataFrame":
    """Return counts dictionaries as a :class:`pandas.DataFrame`.

    Column names follow the ``c_<bitstring>`` convention and the ``step_index``
    column records the order of the prefixes.
    """

    prefixed = [{f"c_{k}": int(v) for k, v in row.items()} for row in rows]
    return _sequence_of_dicts_to_dataframe(prefixed, index_label="step_index")
