"""Normalization helpers for probability dictionaries."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

try:  # pragma: no cover - numpy is an optional runtime dependency during import
    import numpy as _np
except Exception:  # pragma: no cover - allow import to succeed without numpy present
    _np = None  # type: ignore[assignment]

__all__ = [
    "canonicalize_bitstring_key",
    "normalize_probability_dict",
]

_BINARY_CHARS = {"0", "1"}


def _is_numpy_bool(value: Any) -> bool:
    """Return ``True`` if ``value`` is a NumPy boolean scalar."""

    return _np is not None and isinstance(value, _np.bool_)


def _apply_width(bitstring: str, width: Optional[int]) -> str:
    """Return ``bitstring`` padded/validated against ``width``.

    Args:
        bitstring: Binary string (already stripped of whitespace and underscores).
        width: Desired output width in bits. ``None`` leaves the string unchanged.

    Returns:
        str: Bitstring with the requested width.

    Raises:
        ValueError: If ``bitstring`` is incompatible with ``width``.
    """

    if width is None:
        return bitstring
    if width == 0:
        if bitstring.strip("0"):
            raise ValueError("Encountered non-zero bitstring for a 0-qubit probability dictionary.")
        return ""
    if len(bitstring) < width:
        return bitstring.rjust(width, "0")
    if len(bitstring) > width:
        raise ValueError(
            f"Bitstring '{bitstring}' has length {len(bitstring)} but expected {width} bits."
        )
    return bitstring


def _clean_string_bits(text: str, width: Optional[int]) -> str:
    cleaned = "".join(text.split()).replace("_", "")
    if cleaned:
        invalid = set(cleaned) - _BINARY_CHARS
        if invalid:
            raise ValueError(
                f"Bitstring '{text}' contains non-binary characters: {sorted(invalid)!r}."
            )
    return _apply_width(cleaned, width)


def _sequence_to_bitstring(bits: Sequence[Any], width: Optional[int]) -> str:
    chars: list[str] = []
    for element in bits:
        if isinstance(element, (str, bytes)):
            text = element.decode("utf-8", "replace") if isinstance(element, bytes) else element
            cleaned = "".join(text.split()).replace("_", "")
            if not cleaned:
                raise ValueError("Bitstring sequence elements must not be empty strings.")
            if set(cleaned) - _BINARY_CHARS:
                raise ValueError(
                    f"Bitstring sequence elements must be binary digits; got {element!r}."
                )
            chars.extend(cleaned)
            continue
        try:
            value = int(element)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot interpret bitstring element {element!r} as an integer."  # pragma: no cover
            ) from exc
        if value not in (0, 1):
            raise ValueError(f"Bitstring sequence entries must be 0 or 1; got {element!r}.")
        chars.append("1" if value else "0")
    bitstring = "".join(chars)
    return _apply_width(bitstring, width)


def _int_to_bitstring(value: int, width: Optional[int]) -> str:
    if value < 0:
        raise ValueError("Probability dictionary basis indices must be non-negative integers.")
    if width == 0:
        if value != 0:
            raise ValueError(
                "Non-zero basis index encountered for a 0-qubit probability dictionary."
            )
        return ""
    if width is None:
        bitstring = format(value, "b")
    else:
        bitstring = format(value, f"0{width}b")
    return _apply_width(bitstring, width)


def _infer_width_from_key(key: Any) -> Optional[int]:
    if isinstance(key, str):
        cleaned = "".join(key.split()).replace("_", "")
        return len(cleaned)
    if isinstance(key, bytes):
        return _infer_width_from_key(key.decode("utf-8", "replace"))
    if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        try:
            return len(key)
        except TypeError:  # pragma: no cover - defensive
            return None
    try:
        value = operator.index(key)
    except TypeError:
        if _is_numpy_bool(key):
            value = int(key)
        else:
            return None
    if value == 0:
        return 1
    return value.bit_length()


def _normalize_key(key: Any, width: Optional[int]) -> str:
    if isinstance(key, str):
        return _clean_string_bits(key, width)
    if isinstance(key, bytes):
        return _clean_string_bits(key.decode("utf-8", "replace"), width)
    if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        return _sequence_to_bitstring(key, width)
    try:
        return _int_to_bitstring(operator.index(key), width)
    except TypeError:
        if _is_numpy_bool(key):
            return _int_to_bitstring(int(key), width)
    # Last resort: coerce to string and validate
    return _clean_string_bits(str(key), width)


def _coerce_optional_num_qubits(width: Optional[Any]) -> Optional[int]:
    if width is None:
        return None
    try:
        coerced = operator.index(width)
    except TypeError as exc:
        if _is_numpy_bool(width):
            coerced = int(width)
        else:
            raise TypeError("num_qubits must be an integer when specified.") from exc
    if coerced < 0:
        raise ValueError("num_qubits must be non-negative when specified.")
    return int(coerced)


def canonicalize_bitstring_key(key: Any, *, width: Optional[int] = None) -> str:
    """Return ``key`` as a canonical binary string label."""

    return _normalize_key(key, width)


def normalize_probability_dict(
    prob_dict: Mapping[Any, Any], *, num_qubits: Optional[int] = None
) -> Dict[str, float]:
    """Return a probability dictionary with canonical string bitstring keys.

    Args:
        prob_dict: Mapping produced by ``Statevector.probabilities_dict`` or similar
            helpers.
        num_qubits: Optional explicit number of qubits represented by the distribution.
            When provided, keys are validated to match this width and padded with leading
            zeros if required.

    Returns:
        dict[str, float]: Normalized probabilities with stable, lexicographically
        sorted string keys and rounded float values. Entries that normalize to the
        same bitstring are summed together before rounding.

    Raises:
        TypeError: If ``prob_dict`` is not a mapping.
        ValueError: If keys cannot be interpreted as binary strings with the
            requested width or if any probability value is non-finite or
            significantly negative.
    """

    if not isinstance(prob_dict, Mapping):
        raise TypeError("Probability data must be a mapping of basis states to values.")

    width = num_qubits
    if width is None:
        inferred = [
            w for w in (_infer_width_from_key(k) for k in prob_dict.keys()) if w is not None
        ]
        width = max(inferred) if inferred else None
    else:
        width = _coerce_optional_num_qubits(width)

    accumulated: Dict[str, float] = {}
    for raw_key, value in prob_dict.items():
        key = canonicalize_bitstring_key(raw_key, width=width)
        try:
            prob = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Probability for key '{key}' must be a real number; got {value!r}."
            ) from exc
        if not math.isfinite(prob):
            raise ValueError("Probability values must be finite real numbers.")
        if prob < 0.0:
            if prob > -1e-15:
                prob = 0.0
            else:
                raise ValueError(f"Probability value for key '{key}' is negative ({prob}).")
        accumulated[key] = math.fsum((accumulated.get(key, 0.0), prob))

    normalized: Dict[str, float] = {}
    for key in sorted(accumulated):
        prob = accumulated[key]
        rounded = round(prob, 15)
        stored = rounded if rounded != 0.0 or prob == 0.0 else prob
        if stored == 0.0:
            stored = 0.0
        normalized[key] = stored
    if width == 0 and not normalized:
        # ``width`` reflects the requested number of qubits. When zero and the input mapping
        # contains no explicit entries, ``normalized`` would otherwise be empty even though the
        # only valid basis state should have probability 1. Return that explicit distribution so
        # callers always receive a well-defined probability dictionary.
        return {"": 1.0}
    return normalized
