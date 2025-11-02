"""High-level analytics helpers for prefix probability data."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any, Callable, List, Optional

from qiskit import QuantumCircuit  # type: ignore[import-untyped]
from qiskit.quantum_info import (  # type: ignore[import-untyped]
    hellinger_distance as _qi_hellinger_distance,
    shannon_entropy as _qi_shannon_entropy,
)

from .backend_trace import (
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
)
from .probabilities import normalize_probability_dict

__all__ = [
    "shannon_entropy",
    "prefix_shannon_entropies",
    "trace_shannon_entropy_with_statevector",
    "trace_shannon_entropy_with_sampler",
    "total_variation_distance",
    "cross_entropy",
    "kullback_leibler_divergence",
    "jensen_shannon_divergence",
    "hellinger_distance",
    "prefix_total_variation_distances",
    "prefix_cross_entropies",
    "prefix_kullback_leibler_divergences",
    "prefix_jensen_shannon_divergences",
    "prefix_hellinger_distances",
    "trace_total_variation_distance_with_statevector",
    "trace_total_variation_distance_with_sampler",
    "trace_cross_entropy_with_statevector",
    "trace_cross_entropy_with_sampler",
    "trace_kullback_leibler_divergence_with_statevector",
    "trace_kullback_leibler_divergence_with_sampler",
    "trace_jensen_shannon_divergence_with_statevector",
    "trace_jensen_shannon_divergence_with_sampler",
    "trace_hellinger_distance_with_statevector",
    "trace_hellinger_distance_with_sampler",
]


def _entropy_log_base(base: float) -> float:
    """Return ``log(base)`` after validating ``base`` for entropy calculations."""

    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("Entropy log base must be positive and not equal to 1.")
    return math.log(base)


def _normalized_probability_distribution(
    data: Mapping[Any, Any], *, num_qubits: Optional[int] = None
) -> dict[str, float]:
    """Return ``data`` as a normalized probability distribution."""

    normalized = normalize_probability_dict(data, num_qubits=num_qubits)
    if not normalized:
        raise ValueError("Probability distribution must not be empty.")

    total = math.fsum(normalized.values())
    if total <= 0.0:
        raise ValueError("Probability distribution must sum to a positive value.")

    scale = 1.0 / total
    return {key: value * scale for key, value in normalized.items()}


def _aligned_probability_distributions(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    num_qubits: Optional[int] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return two normalized distributions with validated bitstring widths."""

    p = _normalized_probability_distribution(first, num_qubits=num_qubits)
    q = _normalized_probability_distribution(second, num_qubits=num_qubits)

    width_p = {len(key) for key in p}
    width_q = {len(key) for key in q}
    if len(width_p) != 1 or len(width_q) != 1:
        raise ValueError(
            "Normalized probability dictionaries must have consistent bitstring widths."
        )
    if width_p != width_q:
        raise ValueError("Probability distributions must share the same bitstring width.")

    return p, q


def _resolve_width_per_prefix(
    num_qubits: Optional[int | Sequence[int]], count: int
) -> List[Optional[int]]:
    if num_qubits is None:
        return [None] * count
    if isinstance(num_qubits, Integral):
        width = int(num_qubits)
        if width < 0:
            raise ValueError("num_qubits must be non-negative when specified.")
        return [width] * count
    if isinstance(num_qubits, Sequence) and not isinstance(num_qubits, (str, bytes)):
        values = list(num_qubits)
        widths: List[Optional[int]] = []
        for entry in values[:count]:
            if entry is None:
                widths.append(None)
                continue
            if isinstance(entry, Integral):
                width = int(entry)
                if width < 0:
                    raise ValueError("num_qubits entries must be non-negative when specified.")
                widths.append(width)
                continue
            raise TypeError("num_qubits sequence entries must be integers or None.")
        if len(widths) < count:
            widths.extend([None] * (count - len(widths)))
        return widths
    raise TypeError("num_qubits must be None, an int, or a sequence of ints.")


def _resolve_reference_per_prefix(
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    count: int,
) -> List[Optional[Mapping[Any, Any]]]:
    """Return a list of reference distributions aligned to ``count`` prefixes."""

    if isinstance(reference, Mapping):
        return [reference] * count

    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        refs: List[Optional[Mapping[Any, Any]]] = []
        for entry in list(reference)[:count]:
            if entry is None:
                refs.append(None)
                continue
            if isinstance(entry, Mapping):
                refs.append(entry)
                continue
            raise TypeError("Reference sequence entries must be mappings or None.")
        if len(refs) < count:
            raise ValueError("Reference sequence must supply an entry for every prefix.")
        return refs

    raise TypeError("reference must be a mapping or sequence of mappings.")


def _metric_per_prefix(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    metric: Callable[..., float],
    *,
    num_qubits: Optional[int | Sequence[int]] = None,
    base: Optional[float] = None,
) -> List[float]:
    widths = _resolve_width_per_prefix(num_qubits, len(prefixes))
    refs = _resolve_reference_per_prefix(reference, len(prefixes))
    results: List[float] = []
    for index, (prefix, ref, width) in enumerate(zip(prefixes, refs, widths)):
        if ref is None:
            results.append(math.nan)
            continue
        kwargs = {"num_qubits": width}
        if base is not None:
            kwargs["base"] = base
        try:
            results.append(metric(prefix, ref, **kwargs))
        except ValueError as exc:
            raise ValueError(f"{exc} (at prefix index {index})") from exc
    return results


def shannon_entropy(
    probabilities: Mapping[Any, Any], *, base: float = 2.0, num_qubits: Optional[int] = None
) -> float:
    """Return the Shannon entropy of ``probabilities``.

    Args:
        probabilities: Probability dictionary or counts mapping. Keys are coerced
            to canonical bitstring labels and values are normalized prior to
            computing the entropy.
        base: Logarithm base. Defaults to ``2`` which yields entropy measured in
            bits. Any positive base except ``1`` is accepted.
        num_qubits: Optional explicit number of qubits represented by the
            distribution. When provided, bitstrings are validated to match the
            requested width.

    Returns:
        float: Shannon entropy ``-sum(p * log(p))`` using the requested base.
    """

    _entropy_log_base(base)
    normalized = _normalized_probability_distribution(probabilities, num_qubits=num_qubits)

    # Delegate the final entropy calculation to Qiskit's reference implementation.
    # This keeps ``qiskit_inspect`` aligned with the core library while our
    # normalization still provides the flexible input handling (counts, quasi
    # distributions, dicts, etc.).
    values = list(normalized.values())
    return float(_qi_shannon_entropy(values, base=base))


def prefix_shannon_entropies(
    prefixes: Sequence[Mapping[Any, Any]],
    *,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return the Shannon entropy for each prefix probability dictionary.

    ``num_qubits`` may be an integer applied to every prefix or a sequence
    providing the classical width for each entry individually. When omitted the
    width is inferred from the keys in ``prefixes``.
    """

    widths = _resolve_width_per_prefix(num_qubits, len(prefixes))
    entropies: List[float] = []
    for probabilities, width in zip(prefixes, widths):
        entropies.append(shannon_entropy(probabilities, base=base, num_qubits=width))
    return entropies


def trace_shannon_entropy_with_statevector(
    circuit: QuantumCircuit,
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
) -> List[float]:
    """Compute Shannon entropy for each prefix using exact statevector tracing."""

    probabilities = trace_probabilities_with_statevector_exact(
        circuit,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )
    return prefix_shannon_entropies(probabilities, base=base)


def trace_shannon_entropy_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
) -> List[float]:
    """Compute Shannon entropy for each prefix using a SamplerV2-compatible backend."""

    probabilities = trace_probabilities_with_sampler(
        circuit,
        sampler,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
    )
    return prefix_shannon_entropies(probabilities, base=base)


def _metric_series_from_probabilities(
    probabilities: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    metric: Callable[..., List[float]],
    *,
    num_qubits: Optional[int | Sequence[int]] = None,
    base: Optional[float] = None,
) -> List[float]:
    kwargs: dict[str, Any] = {}
    if num_qubits is not None:
        kwargs["num_qubits"] = num_qubits
    if base is not None:
        kwargs["base"] = base
    return metric(probabilities, reference, **kwargs)


def _trace_metric_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    metric: Callable[..., List[float]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
    base: Optional[float] = None,
) -> List[float]:
    probabilities = trace_probabilities_with_statevector_exact(
        circuit,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )
    return _metric_series_from_probabilities(
        probabilities,
        reference,
        metric,
        num_qubits=num_qubits,
        base=base,
    )


def _trace_metric_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    metric: Callable[..., List[float]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
    base: Optional[float] = None,
) -> List[float]:
    probabilities = trace_probabilities_with_sampler(
        circuit,
        sampler,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
    )
    return _metric_series_from_probabilities(
        probabilities,
        reference,
        metric,
        num_qubits=num_qubits,
        base=base,
    )


def total_variation_distance(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    num_qubits: Optional[int] = None,
) -> float:
    """Return the total-variation distance between two distributions."""

    p, q = _aligned_probability_distributions(first, second, num_qubits=num_qubits)

    support = set(p) | set(q)
    deltas = [abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in support]
    return 0.5 * math.fsum(deltas)


def cross_entropy(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    base: float = 2.0,
    num_qubits: Optional[int] = None,
) -> float:
    """Return the cross entropy ``H(first, second)`` in the requested base."""

    log_base = _entropy_log_base(base)
    p, q = _aligned_probability_distributions(first, second, num_qubits=num_qubits)

    contributions = []
    for key, prob in p.items():
        if prob <= 0.0:
            continue
        q_prob = q.get(key, 0.0)
        if q_prob <= 0.0:
            raise ValueError(
                "Cross entropy is undefined when the second distribution assigns zero probability to a non-zero outcome in the first distribution."
            )
        contributions.append(prob * (math.log(q_prob) / log_base))
    return -math.fsum(contributions)


def _kullback_leibler_divergence_aligned(
    p: Mapping[str, float],
    q: Mapping[str, float],
    *,
    log_base: float,
) -> float:
    contributions = []
    for key, prob in p.items():
        if prob <= 0.0:
            continue
        q_prob = q.get(key, 0.0)
        if q_prob <= 0.0:
            raise ValueError(
                "Kullback-Leibler divergence is undefined when the second distribution assigns zero probability to a non-zero outcome in the first distribution."
            )
        contributions.append(prob * (math.log(prob / q_prob) / log_base))
    return math.fsum(contributions)


def kullback_leibler_divergence(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    base: float = 2.0,
    num_qubits: Optional[int] = None,
) -> float:
    """Return the Kullback-Leibler divergence ``D_KL(first || second)``."""

    log_base = _entropy_log_base(base)
    p, q = _aligned_probability_distributions(first, second, num_qubits=num_qubits)
    return _kullback_leibler_divergence_aligned(p, q, log_base=log_base)


def jensen_shannon_divergence(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    base: float = 2.0,
    num_qubits: Optional[int] = None,
) -> float:
    """Return the Jensen-Shannon divergence between two distributions."""

    log_base = _entropy_log_base(base)
    p, q = _aligned_probability_distributions(first, second, num_qubits=num_qubits)

    support = set(p) | set(q)
    midpoint = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in support}
    # ``midpoint`` is already normalized because both inputs sum to one.
    return 0.5 * (
        _kullback_leibler_divergence_aligned(p, midpoint, log_base=log_base)
        + _kullback_leibler_divergence_aligned(q, midpoint, log_base=log_base)
    )


def hellinger_distance(
    first: Mapping[Any, Any],
    second: Mapping[Any, Any],
    *,
    num_qubits: Optional[int] = None,
) -> float:
    """Return the Hellinger distance between two probability distributions."""

    p, q = _aligned_probability_distributions(first, second, num_qubits=num_qubits)

    # Defer to Qiskit's canonical implementation once we have harmonized the
    # inputs, ensuring we pick up any numerical improvements from upstream.
    return float(_qi_hellinger_distance(p, q))


def prefix_total_variation_distances(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return total-variation distance for each prefix against ``reference``."""

    return _metric_per_prefix(
        prefixes,
        reference,
        total_variation_distance,
        num_qubits=num_qubits,
    )


def prefix_cross_entropies(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return cross entropy for each prefix against ``reference``."""

    _entropy_log_base(base)
    return _metric_per_prefix(
        prefixes,
        reference,
        cross_entropy,
        num_qubits=num_qubits,
        base=base,
    )


def prefix_kullback_leibler_divergences(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return KL divergence for each prefix against ``reference``."""

    _entropy_log_base(base)
    return _metric_per_prefix(
        prefixes,
        reference,
        kullback_leibler_divergence,
        num_qubits=num_qubits,
        base=base,
    )


def prefix_jensen_shannon_divergences(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return Jensen-Shannon divergence for each prefix against ``reference``."""

    _entropy_log_base(base)
    return _metric_per_prefix(
        prefixes,
        reference,
        jensen_shannon_divergence,
        num_qubits=num_qubits,
        base=base,
    )


def prefix_hellinger_distances(
    prefixes: Sequence[Mapping[Any, Any]],
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Return Hellinger distance for each prefix against ``reference``."""

    return _metric_per_prefix(
        prefixes,
        reference,
        hellinger_distance,
        num_qubits=num_qubits,
    )


def trace_total_variation_distance_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix total-variation distance using exact statevectors."""

    return _trace_metric_with_statevector(
        circuit,
        reference,
        prefix_total_variation_distances,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
    )


def trace_total_variation_distance_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix total-variation distance using a sampler backend."""

    return _trace_metric_with_sampler(
        circuit,
        sampler,
        reference,
        prefix_total_variation_distances,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
    )


def trace_cross_entropy_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix cross entropy using exact statevectors."""

    _entropy_log_base(base)
    return _trace_metric_with_statevector(
        circuit,
        reference,
        prefix_cross_entropies,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_cross_entropy_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix cross entropy using a sampler backend."""

    _entropy_log_base(base)
    return _trace_metric_with_sampler(
        circuit,
        sampler,
        reference,
        prefix_cross_entropies,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_kullback_leibler_divergence_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix KL divergence using exact statevectors."""

    _entropy_log_base(base)
    return _trace_metric_with_statevector(
        circuit,
        reference,
        prefix_kullback_leibler_divergences,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_kullback_leibler_divergence_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix KL divergence using a sampler backend."""

    _entropy_log_base(base)
    return _trace_metric_with_sampler(
        circuit,
        sampler,
        reference,
        prefix_kullback_leibler_divergences,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_jensen_shannon_divergence_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix Jensen-Shannon divergence using exact statevectors."""

    _entropy_log_base(base)
    return _trace_metric_with_statevector(
        circuit,
        reference,
        prefix_jensen_shannon_divergences,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_jensen_shannon_divergence_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    base: float = 2.0,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix Jensen-Shannon divergence using a sampler backend."""

    _entropy_log_base(base)
    return _trace_metric_with_sampler(
        circuit,
        sampler,
        reference,
        prefix_jensen_shannon_divergences,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
        base=base,
    )


def trace_hellinger_distance_with_statevector(
    circuit: QuantumCircuit,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    include_initial: bool = False,
    initial_state: Optional[Any] = None,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix Hellinger distance using exact statevectors."""

    return _trace_metric_with_statevector(
        circuit,
        reference,
        prefix_hellinger_distances,
        include_initial=include_initial,
        initial_state=initial_state,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
    )


def trace_hellinger_distance_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    reference: Mapping[Any, Any] | Sequence[Optional[Mapping[Any, Any]]],
    *,
    shots: int = 4096,
    debug_bit_order: bool = False,
    parameter_values: Optional[Any] = None,
    num_qubits: Optional[int | Sequence[int]] = None,
) -> List[float]:
    """Trace per-prefix Hellinger distance using a sampler backend."""

    return _trace_metric_with_sampler(
        circuit,
        sampler,
        reference,
        prefix_hellinger_distances,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
        num_qubits=num_qubits,
    )
