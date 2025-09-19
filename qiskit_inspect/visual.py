"""Small helpers for presenting states and probability distributions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qiskit.quantum_info import Statevector  # type: ignore[import-untyped]


def pretty_ket(state: Statevector, threshold: float = 1e-3) -> str:
    """Return a sparse Dirac ket string for significant amplitudes.

    Only amplitudes with magnitude ``>= threshold`` are shown. Complex numbers
    are formatted as ``a+bi`` with four decimal places.

    Args:
        state: Statevector to inspect.
        threshold: Magnitude threshold below which amplitudes are omitted.

    Returns:
        str: Human-readable ket expression like ``0.7071|00> + 0.7071|11>``.
    """
    n = state.num_qubits
    vec = state.data
    parts: List[str] = []
    for i, a in enumerate(vec):
        if abs(a) < threshold:
            continue
        real = round(a.real, 4)
        imag = round(a.imag, 4)

        if abs(real) < 5e-5:
            real = 0.0
        if abs(imag) < 5e-5:
            imag = 0.0

        if imag == 0.0:
            coef = f"{real:.4f}"
        elif real == 0.0:
            sign = "-" if imag < 0 else ""
            coef = f"{sign}{abs(imag):.4f}j"
        else:
            sign = "+" if imag >= 0 else "-"
            coef = f"{real:.4f}{sign}{abs(imag):.4f}j"
        basis = format(i, f"0{n}b") if n else ""
        parts.append(f"{coef}|{basis}>")

    if not parts:
        return "0"

    formatted: List[str] = []
    for idx, part in enumerate(parts):
        if idx == 0:
            formatted.append(part)
        elif part.startswith("-"):
            formatted.append(f"- {part[1:]}")
        else:
            formatted.append(f"+ {part}")
    return " ".join(formatted)


def top_amplitudes(state: Statevector, k: int = 8) -> List[Tuple[str, complex]]:
    """Return the top-``k`` amplitudes by magnitude.

    Args:
        state: Statevector to inspect.
        k: Number of entries to return.

    Returns:
        list[tuple[str, complex]]: Pairs of bitstring and complex amplitude.
    """
    n = state.num_qubits
    vec = state.data
    ranked = sorted(((i, a) for i, a in enumerate(vec)), key=lambda t: abs(t[1]), reverse=True)
    return [(format(i, f"0{n}b"), a) for i, a in ranked[:k]]


def ascii_histogram(probs: Dict[str, float], width: int = 40, sort: bool = True) -> str:
    """Return a small ASCII bar chart of a probability dict.

    Args:
        probs: Mapping of bitstring to probability.
        width: Maximum bar width in characters.
        sort: If ``True``, sort bars by descending probability.

    Returns:
        str: Multiline string with one bar per bitstring.
    """
    if not probs:
        return ""
    # Normalize to avoid >1 totals; keep relative scale.
    total = sum(probs.values()) or 1.0
    norm = probs if abs(total - 1.0) < 1e-12 else {k: v / total for k, v in probs.items()}
    items = list(norm.items())
    if sort:
        items.sort(key=lambda kv: kv[1], reverse=True)
    maxv = max((v for _, v in items), default=1.0) or 1.0
    lines: List[str] = []
    for k, v in items:
        n = int(round((v / maxv) * width)) if maxv > 0 else 0
        lines.append(f"{k}: {'#' * n} {v:.4f}")
    return "\n".join(lines)


def format_classical_bits(bits: List[Optional[int]], unknown: str = "x") -> str:
    """Return classical bits in circuit order, using ``unknown`` for ``None``.

    Args:
        bits: Classical bit values (0/1 or ``None``) in circuit order.
        unknown: Placeholder character for unknown values.

    Returns:
        str: Compact string like ``x10`` for three classical bits.
    """
    return "".join(unknown if b is None else str(int(b)) for b in bits)
