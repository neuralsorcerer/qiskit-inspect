"""Export utilities for expectations and execution traces."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from .debugger import TraceRecord
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


def _ensure_trace_dicts(
    records: Iterable[Union[TraceRecord, dict]], state_format: str
) -> List[dict]:
    """Convert a sequence of ``TraceRecord``/dict into a list of dicts.

    Args:
        records: Iterable of :class:`TraceRecord` or already-serialized dicts.
        state_format: Passed to :meth:`TraceRecord.to_dict` when serializing.

    Returns:
        list[dict]: Plain dicts representing trace records.
    """
    out: List[dict] = []
    for r in records:
        if isinstance(r, TraceRecord):
            out.append(r.to_dict(state_format=state_format))
        else:
            out.append(r)
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


def _coerce_classical_bits(value: object) -> List[Optional[int]]:
    """Return ``value`` as a list of optional integers describing classical bits."""

    if value is None:
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
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if item is None:
                out.append(None)
            elif isinstance(item, (bool, int)):
                out.append(int(item))
            elif isinstance(item, str) and item.strip() in {"0", "1"}:
                out.append(int(item.strip()))
            else:
                raise TypeError(
                    "Classical bit entries must be 0, 1, None, or strings containing 0/1."
                )
        return out
    raise TypeError("Classical bit values must be provided as a sequence or string.")


def write_trace_csv(
    records: Iterable[Union[TraceRecord, dict]], file_path: str, state_format: str = "probs"
) -> None:
    """Write trace snapshots to CSV.

    Args:
        records: Iterable of :class:`TraceRecord` or dicts.
        file_path: Destination CSV path.
        state_format: Only ``"probs"`` is supported for CSV (amplitudes are not
            supported in this format).

    Raises:
        ValueError: If ``state_format`` is not ``"probs"``.

    Notes:
        Columns: ``step_index``, ``instruction``, ``classical_bits`` (string),
        followed by probability columns named ``p_<bitstring>``.
    """
    if state_format != "probs":
        raise ValueError("write_trace_csv supports only state_format='probs'")
    dicts = _ensure_trace_dicts(records, state_format=state_format)
    # Union of probability keys
    pkeys: Set[str] = set()
    for d in dicts:
        p = d.get("state", {}) or {}
        pkeys.update(p.keys())
    pcols = [f"p_{k}" for k in sorted(pkeys)]
    header = ["step_index", "instruction", "classical_bits"] + pcols
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in dicts:
            step = d.get("step_index")
            instr = d.get("instruction")
            cbits = d.get("classical_bits", [])
            cbits_str = format_classical_bits(cbits)
            p = d.get("state", {}) or {}
            row = [step, instr, cbits_str] + [p.get(k[2:], 0.0) for k in pcols]
            w.writerow(row)


def write_trace_json(
    records: Iterable[Union[TraceRecord, dict]], file_path: str, state_format: str = "probs"
) -> None:
    """Write trace snapshots to a JSON file.

    If ``records`` contains :class:`TraceRecord` objects, they are serialized using
    the requested ``state_format`` before writing.

    Args:
        records: Iterable of :class:`TraceRecord` or dicts.
        file_path: Destination JSON path.
        state_format: Serialization format for the state (``"probs"`` or
            ``"amplitudes"``).
    """
    dicts = _ensure_trace_dicts(records, state_format=state_format)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2)


def trace_records_to_dataframe(
    records: Iterable[Union[TraceRecord, dict]],
    *,
    state_format: str = "probs",
    classical_bits: bool = True,
    classical_bit_columns: bool | Sequence[str] = False,
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

    Returns:
        pandas.DataFrame: A data frame with one row per trace record ordered as
        provided.
    """

    pd = _require_pandas()

    dicts = _ensure_trace_dicts(records, state_format=state_format)
    base_columns: List[str] = ["step_index", "instruction"]

    state_columns: List[str]
    flattened_states: List[Dict[str, Union[float, complex]]] = []
    if state_format == "probs":
        all_keys: Set[str] = set()
        flattened: List[Dict[str, Union[float, complex]]] = []
        for d in dicts:
            probs = d.get("state", {}) or {}
            entry: Dict[str, Union[float, complex]] = {
                f"p_{key}": float(value) for key, value in probs.items()
            }
            flattened.append(entry)
            all_keys.update(entry.keys())
        state_columns = sorted(all_keys)
        flattened_states = flattened
    elif state_format == "amplitudes":
        max_len = 0
        flattened_complex: List[Dict[str, complex]] = []
        for d in dicts:
            amps = d.get("state", []) or []
            amp_entry: Dict[str, complex] = {}
            for idx, pair in enumerate(amps):
                real_imag: Sequence[float] = pair
                if not isinstance(real_imag, Sequence) or len(real_imag) != 2:
                    raise ValueError("Amplitude entries must be [re, im] pairs.")
                amp_entry[f"amp_{idx}"] = complex(float(real_imag[0]), float(real_imag[1]))
            flattened_complex.append(amp_entry)
            max_len = max(max_len, len(amps))
        state_columns = [f"amp_{i}" for i in range(max_len)]
        flattened_states = flattened_complex
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
    columns = base_columns + bit_column_names + state_columns

    rows: List[List[Optional[Union[int, str, float, complex]]]] = []
    for d, cbits, state_values in zip(dicts, classical_lists, flattened_states):
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
