"""Test-time assertions for quantum states and distributions."""

from __future__ import annotations

from typing import Dict

from qiskit.quantum_info import Statevector  # type: ignore[import-untyped]


def assert_state_equiv(
    a: Statevector, b: Statevector, rtol: float = 1e-6, atol: float = 1e-9
) -> None:
    """Assert two statevectors are equivalent up to a global phase.

    Uses :meth:`qiskit.quantum_info.Statevector.equiv` under the hood.

    Args:
        a: First statevector.
        b: Second statevector.
        rtol: Relative tolerance passed to ``equiv``.
        atol: Absolute tolerance passed to ``equiv``.

    Raises:
        AssertionError: If the states are not equivalent up to a global phase.
    """
    if not a.equiv(b, rtol=rtol, atol=atol):
        raise AssertionError("Statevectors are not equivalent up to global phase.")


def _l1(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Return L1 distance between two probability-like mappings.

    Missing keys are treated as zeros. The input values are not normalized.
    """
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def assert_probabilities_close(
    observed: Dict[str, float],
    expected: Dict[str, float],
    tol_l1: float = 0.02,
) -> None:
    """Assert two probability distributions are close in L1 distance.

    Args:
        observed: Mapping of bitstring to probability.
        expected: Mapping of bitstring to probability.
        tol_l1: Maximum allowed L1 distance (sum of absolute differences).

    Raises:
        AssertionError: If the L1 distance exceeds ``tol_l1``.
    """
    d = _l1(observed, expected)
    if d > tol_l1:
        raise AssertionError(f"Probability distributions differ (L1={d:.4f} > {tol_l1:.4f}).")


def assert_counts_close(
    observed_counts: Dict[str, int],
    expected_probs: Dict[str, float],
    shots: int,
    tol_l1: float = 0.03,
) -> None:
    """Assert sampled counts are close to expected probabilities.

    Converts counts into probabilities using ``shots`` (or the sum of counts if
    ``shots`` is ``0``/``None``) and then delegates to
    :func:`assert_probabilities_close`.

    Args:
        observed_counts: Mapping of bitstring to observed counts.
        expected_probs: Mapping of bitstring to expected probabilities.
        shots: Number of shots used to collect ``observed_counts`` (used for
            normalization). If falsy, the sum of counts is used.
        tol_l1: Maximum allowed L1 distance between the normalized distributions.

    Raises:
        AssertionError: If the L1 distance exceeds ``tol_l1``.
    """
    total = shots if shots else max(1, sum(observed_counts.values()))
    observed_probs = {k: v / total for k, v in observed_counts.items()}
    assert_probabilities_close(observed_probs, expected_probs, tol_l1=tol_l1)
