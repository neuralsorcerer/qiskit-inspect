"""Utilities for normalizing sampler outputs into concrete count dictionaries."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from decimal import Decimal
from numbers import Integral, Rational, Real
from typing import Any, Dict, Sequence, Tuple

from qiskit.exceptions import QiskitError
from qiskit.result import utils as result_utils

from .probabilities import canonicalize_bitstring_key

__all__ = [
    "coerce_count_value",
    "coerce_counts",
    "extract_counts",
    "marginalize_counts",
    "extract_total_shots",
]


def coerce_count_value(value: Any) -> int:
    """Return ``value`` as a validated integer.

    Numeric inputs must represent whole numbers. Floats with fractional parts or
    rational values whose denominator is not ``1`` are rejected so that
    accidental truncation cannot silently corrupt counts. Strings (or bytes)
    containing decimal integers are accepted for convenience. ``bool`` values
    are treated as ``0``/``1``.
    """

    # Fast path: genuine integers (including ``bool`` and numpy scalar integrals).
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)

    # Decimal is not registered with :class:`numbers.Real`, so handle it explicitly.
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise TypeError("Counts values must be integers or castable to integers.")
        return int(value)

    # Exact rationals such as ``fractions.Fraction`` – only allow whole numbers.
    if isinstance(value, Rational):
        numerator = int(value.numerator)
        denominator = int(value.denominator)
        if denominator == 0 or numerator % denominator != 0:
            raise TypeError("Counts values must be integers or castable to integers.")
        return numerator // denominator

    # Real numbers (floats, Decimal, numpy floats) must be integral.
    if isinstance(value, Real):
        float_val = float(value)
        if not float_val.is_integer():
            raise TypeError("Counts values must be integers or castable to integers.")
        return int(round(float_val))

    # Textual representations.
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("Counts values must be integers or castable to integers.") from exc
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypeError("Counts values must be integers or castable to integers.")
        try:
            return int(text)
        except ValueError as exc:
            raise TypeError("Counts values must be integers or castable to integers.") from exc

    # Fallback: rely on ``__int__`` if present.
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError("Counts values must be integers or castable to integers.") from exc
    return coerced


def coerce_counts(counts: Any) -> Dict[str, int]:
    """Return ``counts`` as a plain ``dict[str, int]``.

    Args:
        counts: Mapping or iterable of ``(key, value)`` pairs whose values can be
            converted to integers. Strings and bytes are rejected so they are not
            misinterpreted as iterables of characters.

    Raises:
        TypeError: If ``counts`` is not a mapping or iterable of key/value pairs
            or contains values that cannot be interpreted as integers without
            loss of information.
        ValueError: If a count value is negative.
    """

    items: Iterable[Tuple[Any, Any]]
    if isinstance(counts, Mapping):
        items = counts.items()
    else:
        if isinstance(counts, (str, bytes)):
            raise TypeError("Counts must be a mapping or iterable of key/value pairs.")
        try:
            items = dict(counts).items()
        except (TypeError, ValueError) as exc:
            raise TypeError("Counts must be a mapping or iterable of key/value pairs.") from exc

    bitstring_entries: list[tuple[Any, str, int, bool]] = []
    fallback_entries: list[tuple[str, int]] = []

    for key, value in items:
        coerced_value = coerce_count_value(value)
        if coerced_value < 0:
            raise ValueError("Counts values must be non-negative integers.")

        try:
            cleaned = canonicalize_bitstring_key(key)
        except (TypeError, ValueError):
            fallback_entries.append((str(key), coerced_value))
        else:
            needs_padding = not isinstance(key, (str, bytes, Sequence))
            bitstring_entries.append((key, cleaned, coerced_value, needs_padding))

    coerced: Dict[str, int] = {}

    if bitstring_entries:
        explicit_widths = [len(cleaned) for _, cleaned, _, pad in bitstring_entries if not pad]
        padded_widths = [len(cleaned) for _, cleaned, _, pad in bitstring_entries if pad]
        if explicit_widths:
            inferred_width = max(explicit_widths)
        elif padded_widths:
            inferred_width = max(padded_widths)
        else:
            inferred_width = 0

        for raw_key, cleaned, value, pad in bitstring_entries:
            if pad:
                canonical = canonicalize_bitstring_key(raw_key, width=inferred_width)
            else:
                canonical = cleaned
            coerced[canonical] = coerced.get(canonical, 0) + value

    for label, value in fallback_entries:
        coerced[label] = coerced.get(label, 0) + value

    return coerced


def _counts_from_container(container: Any) -> Dict[str, int] | None:
    """Return counts from a container exposing a ``get_counts``-like API."""

    if container is None:
        return None
    for attr in ("get_counts", "_get_counts", "get_int_counts"):
        getter = getattr(container, attr, None)
        if callable(getter):
            return coerce_counts(getter())
    return None


def _coerce_total_shots(value: Any, *, _visited: set[int] | None = None) -> int | None:
    """Return the non-negative total shots encoded by ``value`` if possible."""

    if value is None:
        return None

    if _visited is None:
        _visited = set()
    marker = id(value)
    if marker in _visited:
        return None
    _visited.add(marker)

    def _attempt(candidate: Any) -> int | None:
        try:
            total = coerce_count_value(candidate)
        except TypeError:
            return None
        if total < 0:
            raise ValueError("Counts values must be non-negative integers.")
        return total

    total = _attempt(value)
    if total is not None:
        return total

    if isinstance(value, Mapping):
        for key in ("total", "shots", "total_shots", "shot_count"):
            total = _coerce_total_shots(value.get(key), _visited=_visited)
            if total is not None:
                return total
        return None

    for attr in ("total", "shots", "total_shots", "shot_count"):
        if hasattr(value, attr):
            total = _coerce_total_shots(getattr(value, attr), _visited=_visited)
            if total is not None:
                return total

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        totals: list[int] = []
        for entry in value:
            total = _coerce_total_shots(entry, _visited=_visited)
            if total is None:
                totals = []
                break
            totals.append(total)
        if totals:
            return sum(totals)

    return None


def extract_total_shots(metadata: Any) -> int | None:
    """Return the non-negative shot total declared in sampler metadata."""

    if metadata is None:
        return None

    candidates: list[Any] = [metadata]
    if isinstance(metadata, Mapping):
        for key in ("shots", "total_shots", "total", "shot_count"):
            candidates.append(metadata.get(key))
    for attr in ("shots", "total_shots", "total", "shot_count"):
        if hasattr(metadata, attr):
            candidates.append(getattr(metadata, attr))

    for candidate in candidates:
        total = _coerce_total_shots(candidate)
        if total is not None:
            return total
    return None


def _counts_from_metadata(pub_res: Any) -> Dict[str, int] | None:
    """Return counts derived from metadata when no classical data is present."""

    metadata = getattr(pub_res, "metadata", None)
    total = extract_total_shots(metadata)
    if total is None:
        return None
    return {"": total}


def _candidate_containers(*objects: Any) -> Iterable[Any]:
    """Yield potential containers that may store counts data."""

    queue = deque(objects)
    seen: set[int] = set()
    while queue:
        candidate = queue.popleft()
        if candidate is None or isinstance(candidate, (str, bytes)):
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        yield candidate

        values_method = getattr(candidate, "values", None)
        if callable(values_method):
            try:
                queue.append(values_method())
            except Exception:  # pragma: no cover - defensive for non-conforming containers
                pass

        if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
            for element in candidate:
                queue.append(element)


def extract_counts(pub_res: Any) -> Dict[str, int]:
    """Best-effort extraction of counts from a SamplerV2 result object."""

    candidates: list[Any] = []

    # Qiskit V2 publication result
    if hasattr(pub_res, "join_data"):
        try:
            data = pub_res.join_data()
        except (TypeError, ValueError):
            data = None
        candidates.append(data)
        values = getattr(data, "values", None)
        if callable(values):
            try:
                candidates.append(values())
            except Exception:  # pragma: no cover - defensive for non-conforming values()
                pass

    # Some variants expose the bit data under ``data.c`` (or ``data.meas``)
    if hasattr(pub_res, "data"):
        data_attr = pub_res.data
        candidates.append(data_attr)
        for name in ("c", "meas"):
            candidates.append(getattr(data_attr, name, None))
        values = getattr(data_attr, "values", None)
        if callable(values):
            try:
                candidates.append(values())
            except Exception:  # pragma: no cover - defensive for non-conforming values()
                pass

    for container in _candidate_containers(*candidates, pub_res):
        counts = _counts_from_container(container)
        if counts is not None:
            return counts

    metadata_counts = _counts_from_metadata(pub_res)
    if metadata_counts is not None:
        return metadata_counts
    raise RuntimeError("Unsupported SamplerV2 result shape: cannot extract counts.")


def marginalize_counts(counts: Dict[str, Any], keep_bits: Sequence[int]) -> Dict[str, Any]:
    """Return counts marginalized onto the specified classical bit positions."""

    indices = list(keep_bits)

    try:
        # Delegate to Qiskit's reference implementation to guarantee identical semantics for
        # well-formed bitstring dictionaries produced by Qiskit itself.
        return dict(result_utils.marginal_counts(counts, indices))
    except QiskitError as exc:
        if not indices:
            raise

        required_width = max(indices) + 1
        if not counts:
            raise

        padded_counts: Dict[str, Any] | None = {}
        for raw_key, value in counts.items():
            try:
                key = canonicalize_bitstring_key(raw_key, width=required_width)
            except (TypeError, ValueError):
                padded_counts = None
                break
            padded_counts[key] = padded_counts.get(key, 0) + value

        if padded_counts is None:
            raise exc

        # Retry with padded keys that meet the requested index width. If Qiskit still
        # rejects the indices we propagate the original error to match its diagnostics.
        return dict(result_utils.marginal_counts(padded_counts, indices))
