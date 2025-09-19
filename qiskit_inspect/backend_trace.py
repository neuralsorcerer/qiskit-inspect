"""Backend prefix tracing and exact expectation helpers."""

from __future__ import annotations

import copy
import inspect
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence as AbcSequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from qiskit import QuantumCircuit  # type: ignore[import-untyped]
from qiskit.exceptions import QiskitError  # type: ignore[import-untyped]
from qiskit.quantum_info import (  # type: ignore[import-untyped]
    Operator,
    Pauli,
    SparsePauliOp,
    Statevector,
)

try:  # pragma: no cover - optional import for runtime type checking
    from qiskit.primitives import StatevectorSampler  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - older Qiskit builds
    StatevectorSampler = None

try:  # pragma: no cover - estimator optional
    from qiskit.primitives import BaseEstimatorV2  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - estimator unavailable
    BaseEstimatorV2 = None  # type: ignore[assignment]

try:  # pragma: no cover - BindingsArray support is optional
    from qiskit.primitives.containers.bindings_array import (  # type: ignore[import-untyped]
        BindingsArray,
    )
except Exception:  # pragma: no cover - BindingsArray unavailable
    BindingsArray = None  # type: ignore[assignment]

from qiskit.primitives.containers.observables_array import (  # type: ignore[import-untyped]
    ObservablesArray,
)

from .debugger import CircuitDebugger
from .prefix_builders import build_prefix_circuits, build_prefix_circuits_for_qubits
from .probabilities import canonicalize_bitstring_key, normalize_probability_dict
from .sampler_results import (
    coerce_count_value,
    extract_counts,
    extract_total_shots,
    marginalize_counts,
)

_LOGGER = logging.getLogger("qiskit_inspect.trace")


def _label_for_clbit(circ: QuantumCircuit, bit) -> str:
    """Return a label for a classical bit as ``reg[index]``.

    Falls back to ``clbit[i]`` if the bit does not belong to a named register.
    """
    try:
        for reg in circ.cregs:
            for i, b in enumerate(reg):
                if b is bit:
                    return f"{reg.name}[{i}]"
    except Exception:
        pass
    idx = circ.find_bit(bit).index
    return f"clbit[{idx}]"


def _format_clbit_order(circ: QuantumCircuit) -> str:
    """Return a diagnostic string showing classical bit index -> label mapping."""
    parts = []
    for i, b in enumerate(circ.clbits):
        parts.append(f"{i}:{_label_for_clbit(circ, b)}")
    return "clbit order: " + ", ".join(parts)


def _prefixes_with_end_measure(circ: QuantumCircuit) -> List[QuantumCircuit]:
    """Compatibility wrapper that builds prefixes via :mod:`prefix_builders`."""

    return build_prefix_circuits(circ)


def _prefixes_with_end_measure_of_qubits(
    circ: QuantumCircuit, qubits: List[int]
) -> List[QuantumCircuit]:
    """Compatibility wrapper that measures only selected qubits when needed."""

    return build_prefix_circuits_for_qubits(circ, qubits)


def _extract_counts(pub_res: Any) -> Dict[str, int]:
    """Backward-compatible wrapper around :func:`sampler_results.extract_counts`."""

    return extract_counts(pub_res)


def _marginalize_counts(counts: Dict[str, int], keep_bits: List[int]) -> Dict[str, int]:
    """Backward-compatible wrapper around :func:`sampler_results.marginalize_counts`."""

    return marginalize_counts(counts, keep_bits)


_TERMINAL_INSTRUCTION_NAMES = {"measure", "barrier", "delay"}

_MidMeasureHelper = Literal["probabilities", "marginals", "counts"]

_MID_MEASURE_HINTS: Dict[_MidMeasureHelper, Tuple[str, str]] = {
    "probabilities": (
        "trace_probabilities_with_sampler",
        "trace_probabilities_with_statevector_exact",
    ),
    "marginals": (
        "trace_marginal_probabilities_with_sampler",
        "trace_marginal_probabilities_with_statevector",
    ),
    "counts": (
        "trace_counts_with_sampler",
        "trace_probabilities_with_statevector_exact",
    ),
}


def _coerce_optional_non_negative_int(value: Any, *, label: str) -> int | None:
    """Return ``value`` as a non-negative ``int`` when specified."""

    if value is None:
        return None

    try:
        coerced = operator.index(value)
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"{label} must be a non-negative integer or None.") from exc

    if coerced < 0:
        raise ValueError(f"{label} must be a non-negative integer or None.")

    return int(coerced)


def _mid_measure_unsupported_message(helper: _MidMeasureHelper, backend_label: str) -> str:
    name, fallback = _MID_MEASURE_HINTS[helper]
    return (
        f"{name} cannot use {backend_label} on circuits containing mid-circuit measurements. "
        f"Use {fallback} for exact results or provide a sampler backend that supports mid-circuit "
        "measurements (for example, qiskit-aer)."
    )


_LOOP_CONTROL_FLOW_NAMES = {"for_loop", "while_loop"}


def _has_mid_circuit_measurement(circ: QuantumCircuit) -> bool:
    """Return ``True`` when ``circ`` measures a qubit and later reuses it."""

    def _walk(instructions, measured: set[int]) -> tuple[bool, set[int]]:
        local = set(measured)
        for inst in instructions:
            op = inst.operation
            blocks = getattr(op, "blocks", None)
            if blocks:
                branch_states: List[set[int]] = []
                for block in blocks:
                    if block is None:
                        continue
                    found, after = _walk(block.data, set(local))
                    if found:
                        return True, set()
                    branch_states.append(after)
                    if op.name in _LOOP_CONTROL_FLOW_NAMES:
                        loop_state = set(after)
                        for _ in range(len(circ.qubits)):
                            loop_found, loop_after = _walk(block.data, set(loop_state))
                            if loop_found:
                                return True, set()
                            if loop_after <= loop_state:
                                break
                            branch_states.append(loop_after)
                            loop_state = loop_after
                if branch_states:
                    merged: set[int] = set()
                    for branch in branch_states:
                        merged.update(branch)
                    local = merged
                continue

            if op.name == "measure":
                for q in inst.qubits:
                    local.add(circ.find_bit(q).index)
                continue

            if op.name in _TERMINAL_INSTRUCTION_NAMES:
                continue

            if not inst.qubits:
                continue

            q_indices = {circ.find_bit(q).index for q in inst.qubits}
            if local & q_indices:
                return True, set()
        return False, local

    found, _ = _walk(circ.data, set())
    return found


def _probabilities_from_statevector(state: Statevector) -> Dict[str, float]:
    """Return a canonical probability dictionary for ``state``."""

    if state.num_qubits == 0:
        # ``Statevector.probabilities_dict`` raises ``ValueError`` when no qubits are
        # present because a basis label cannot be inferred.  Return the lone basis
        # state explicitly so callers still receive a well-defined distribution.
        return {"": 1.0}
    return normalize_probability_dict(state.probabilities_dict(), num_qubits=state.num_qubits)


def _normalize_counts_dict(
    counts: Dict[str, int],
    width: int,
    *,
    total_shots: Optional[int] = None,
) -> Dict[str, int]:
    """Return ``counts`` with canonical bitstring keys while preserving totals."""

    if width < 0:
        raise ValueError("Number of classical bits must be non-negative.")

    declared_total: Optional[int] = None
    if total_shots is not None:
        try:
            declared_total = coerce_count_value(total_shots)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("total_shots must be an integer value.") from exc
        if declared_total < 0:
            raise ValueError("total_shots must be non-negative.")

    aggregated: Dict[str, int] = {}
    total = 0
    for raw_key, raw_value in counts.items():
        key = canonicalize_bitstring_key(raw_key, width=width)
        try:
            value = coerce_count_value(raw_value)
        except TypeError as exc:
            raise TypeError(
                f"Count for key '{key}' must be an integer value; got {raw_value!r}."
            ) from exc
        if value < 0:
            raise ValueError(f"Counts must be non-negative; got {value} for key '{key}'.")
        aggregated[key] = aggregated.get(key, 0) + value
        total += value

    if width == 0:
        inferred_total = total
        if inferred_total == 0 and declared_total is not None:
            inferred_total = declared_total
        if declared_total is not None and inferred_total != declared_total:
            raise ValueError("Provided total_shots does not match the sum of normalized counts.")
        return {"": inferred_total}

    if declared_total is not None and total != declared_total:
        raise ValueError("Provided total_shots does not match the sum of normalized counts.")

    return {key: aggregated.get(key, 0) for key in sorted(aggregated)}


def _measured_clbit_indices(prefix: QuantumCircuit) -> List[int]:
    """Return the sorted indices of classical bits written by measurements."""

    measured: set[int] = set()
    for instruction in prefix.data:
        op = instruction.operation
        if op.name != "measure":
            continue
        if getattr(op, "condition", None) is not None:
            # Conditional measurements may not execute at runtime, so their
            # classical targets do not reliably contain the measured value.
            # Ignore them when determining which classical bits carry sampled
            # results; scratch measurements added by the prefix builder cover
            # the required qubits unconditionally. See Qiskit 2.x measurement
            # semantics: https://docs.quantum.ibm.com/api/qiskit/circuit
            continue
        for cbit in instruction.clbits:
            try:
                measured.add(prefix.find_bit(cbit).index)
            except Exception:  # pragma: no cover - defensive against exotic bit types
                continue
    return sorted(measured)


def _counts_over_measured_clbits(
    prefix: QuantumCircuit,
    counts: Dict[str, int],
) -> tuple[Dict[str, int], int]:
    """Return counts restricted to measured classical bits and their width."""

    measured_indices = _measured_clbit_indices(prefix)
    if measured_indices:
        if measured_indices != list(range(prefix.num_clbits)):
            filtered = marginalize_counts(counts, measured_indices)
            width = len(measured_indices)
        else:
            filtered = counts
            width = prefix.num_clbits
    else:
        total = sum(counts.values())
        filtered = {"": total}
        width = 0
    return filtered, width


def _resolve_effective_total_shots(
    count_total: int, declared_total: int | None, *, width: int
) -> int | None:
    """Return the shot total to enforce when normalizing counts."""

    if declared_total is None:
        return None

    if count_total == 0:
        if width == 0:
            return declared_total
        if declared_total == 0:
            return 0
        raise ValueError(
            "Sampler metadata declared shots but no counts were observed for the measured bits."
        )

    if declared_total == 0:
        raise ValueError("Sampler metadata declared zero shots but non-zero counts were observed.")

    if count_total == declared_total:
        return declared_total

    if declared_total > count_total:
        raise ValueError(
            "Sampler metadata declared more shots than were present in the extracted counts."
        )

    if count_total % declared_total == 0:
        # Parameter broadcasting or batched execution can aggregate several logical runs
        # into a single counts dictionary. In this case the metadata typically reports
        # the per-run shot total. Treat the observed counts as authoritative.
        return None

    raise ValueError("Sampler metadata shots are inconsistent with the extracted counts.")


def _expand_parameter_values(
    parameter_values: Any,
    count: int,
) -> Optional[List[Any]]:
    """Normalize ``parameter_values`` into a list aligned with ``count`` prefixes."""

    if parameter_values is None:
        return None
    if isinstance(parameter_values, Mapping):
        return [parameter_values] * count
    if isinstance(parameter_values, Iterable) and not isinstance(parameter_values, (str, bytes)):
        entries = list(parameter_values)
        if not entries:
            return [entries] * count
        if len(entries) == count:
            return entries
        first = entries[0]
        is_binding = (
            isinstance(first, Mapping)
            or (isinstance(first, Iterable) and not isinstance(first, (str, bytes)))
            or first is None
        )
        if is_binding:
            if len(entries) == 1:
                return [entries[0]] * count
            raise ValueError(
                "parameter_values must provide one entry per prefix when supplying a sequence of bindings."
            )
        return [entries] * count
    return [parameter_values] * count


def _coerce_binding_for_prefix(
    prefix: QuantumCircuit,
    binding: Any,
    base_parameters: Sequence[Any],
) -> Any:
    """Return ``binding`` normalized for the parameters present in ``prefix``."""

    if binding is None:
        return None
    prefix_params = tuple(prefix.parameters)
    if BindingsArray is not None and isinstance(binding, BindingsArray):
        return _bindings_array_to_parameter_mapping(binding, prefix_params, base_parameters)

    if isinstance(binding, Mapping):
        if not prefix_params:
            return {}
        coerced: Dict[Any, Any] = {}
        missing = []
        for param in prefix_params:
            if param in binding:
                coerced[param] = binding[param]
            else:
                missing.append(param)
        if missing:
            raise ValueError(
                "parameter_values are missing assignments for circuit parameters: " f"{missing!r}."
            )
        return coerced
    if isinstance(binding, AbcSequence) and not isinstance(binding, (str, bytes)):
        values = list(binding)
        if len(prefix_params) == len(values) or (not prefix_params and not values):
            return values
        base_params = tuple(base_parameters)
        if base_params and len(values) == len(base_params):
            lookup = {param: val for param, val in zip(base_params, values)}
            if not prefix_params:
                return {}
            coerced = {param: lookup[param] for param in prefix_params if param in lookup}
            if len(coerced) != len(prefix_params):
                missing = [param for param in prefix_params if param not in lookup]
                raise ValueError(
                    "parameter_values are missing assignments for circuit parameters: "
                    f"{missing!r}."
                )
            return coerced
        if not prefix_params:
            if not values:
                return []
            raise ValueError(
                "parameter_values were provided, but the circuit prefix has no parameters."
            )
        raise ValueError(
            "parameter_values must match the number of parameters in the prefix or the "
            "original circuit when provided as a sequence."
        )
    if not prefix_params:
        return {}
    if len(prefix_params) == 1:
        return [binding]
    raise ValueError(
        "parameter_values must be provided as a mapping or sequence when binding "
        "multiple parameters."
    )


def _bindings_array_to_parameter_mapping(
    binding: Any,
    prefix_parameters: Sequence[Any],
    base_parameters: Sequence[Any],
) -> Dict[Any, Any]:
    """Return a mapping of prefix parameters to values from ``binding``."""

    if BindingsArray is None or not isinstance(
        binding, BindingsArray
    ):  # pragma: no cover - defensive
        raise TypeError("BindingsArray support is unavailable in this environment.")

    if not prefix_parameters:
        return {}

    ndim = binding.ndim
    shape = binding.shape

    normalized: Dict[Any, Any] = {}
    base_lookup = {getattr(param, "name", str(param)): param for param in base_parameters}
    prefix_names = {getattr(param, "name", str(param)): param for param in prefix_parameters}

    for key, values in binding.data.items():
        names = key
        array = np.asarray(values)
        for offset, name in enumerate(names):
            param = base_lookup.get(name)
            if param is None:
                raise ValueError(
                    "parameter_values include an assignment for a parameter that is not present in the circuit: "
                    f"'{name}'."
                )
            target = prefix_names.get(name)
            if target is None:
                continue
            slices = (slice(None),) * ndim + (offset,)
            raw = array[slices]
            if ndim == 0:
                value = float(raw)
            else:
                value = np.asarray(raw).reshape(shape)
            normalized[target] = value

    missing = [param for param in prefix_parameters if param not in normalized]
    if missing:
        raise ValueError(
            "parameter_values are missing assignments for circuit parameters: " f"{missing!r}."
        )

    return normalized


def _binding_is_empty(binding: Any) -> bool:
    """Return ``True`` when a binding has no assignments."""

    if isinstance(binding, Mapping):
        return not binding
    if isinstance(binding, (str, bytes)):
        return False
    try:
        return len(binding) == 0
    except TypeError:
        return False


def _build_sampler_pubs(
    prefixes: Sequence[QuantumCircuit], parameter_values: Any, base_parameters: Sequence[Any]
) -> List[Any]:
    """Return sampler publications for ``prefixes`` with optional parameter bindings."""

    normalized = _expand_parameter_values(parameter_values, len(prefixes))
    if not normalized:
        return list(prefixes)
    pubs: List[Any] = []
    for prefix, binding in zip(prefixes, normalized):
        coerced = _coerce_binding_for_prefix(prefix, binding, base_parameters)
        if coerced is None or _binding_is_empty(coerced):
            pubs.append(prefix)
        else:
            pubs.append((prefix, coerced))
    return pubs


def _build_estimator_pubs(
    prefixes: Sequence[QuantumCircuit],
    parameter_values: Any,
    base_parameters: Sequence[Any],
    observables,
) -> List[Any]:
    """Return EstimatorV2 publications for ``prefixes`` and provided observables."""

    normalized = _expand_parameter_values(parameter_values, len(prefixes))
    if normalized is None:
        normalized = [None] * len(prefixes)

    pubs: List[Any] = []
    for prefix, binding in zip(prefixes, normalized):
        if binding is None:
            pubs.append((prefix, observables))
            continue
        coerced = _coerce_binding_for_prefix(prefix, binding, base_parameters)
        if coerced is None or _binding_is_empty(coerced):
            pubs.append((prefix, observables))
        else:
            pubs.append((prefix, observables, coerced))
    return pubs


def _load_aer_sampler_cls():
    """Return the qiskit-aer SamplerV2 class, deferring the import."""

    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # type: ignore[import-untyped]

    return AerSamplerV2


def trace_probabilities_with_sampler(
    circuit: QuantumCircuit,
    sampler,  # StatevectorSampler, Aer SamplerV2, or IBM Runtime SamplerV2
    shots: int | None = 4096,
    debug_bit_order: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, float]]:
    """Sample each instruction prefix with a SamplerV2 and return probabilities.

    Args:
        circuit: Circuit to analyze.
        sampler: A SamplerV2-compatible primitive (for example, Aer SamplerV2,
            IBM Runtime SamplerV2, or a statevector-based sampler).
        shots: Number of shots per prefix. When ``None``, the sampler backend's
            configured default is used.
        debug_bit_order: If ``True``, log classical bit order per prefix for debugging.
        parameter_values: Optional parameter bindings applied to each prefix when
            executing the sampler. Accepts the same formats as
            :meth:`QuantumCircuit.assign_parameters` (mapping or sequence). Provide
            a sequence of bindings with one entry per prefix to vary assignments
            between prefixes.

    Returns:
        list[dict[str, float]]: One probability dictionary per prefix. Keys are
        bitstrings consistent with the circuit's classical bit order for that
        prefix.

    Raises:
        RuntimeError: If counts cannot be extracted from the sampler result object.
    """
    if StatevectorSampler is not None and isinstance(sampler, StatevectorSampler):
        if _has_mid_circuit_measurement(circuit):
            raise RuntimeError(
                _mid_measure_unsupported_message("probabilities", "StatevectorSampler")
            )

    prefixes = _prefixes_with_end_measure(circuit)
    base_parameters = tuple(circuit.parameters)
    pubs = _build_sampler_pubs(prefixes, parameter_values, base_parameters)
    normalized_shots = _coerce_optional_non_negative_int(shots, label="shots")

    run_kwargs: Dict[str, Any] = {}
    if normalized_shots is not None:
        run_kwargs["shots"] = normalized_shots

    try:
        job = sampler.run(pubs, **run_kwargs)
        results = job.result()
    except QiskitError as exc:
        _raise_sampler_mid_measure_error("probabilities", sampler, exc)
        raise  # pragma: no cover - helper always re-raises

    probs_list: List[Dict[str, float]] = []
    for idx, pub_res in enumerate(results):
        prefix = prefixes[idx]
        if debug_bit_order:
            _LOGGER.info(
                "[trace_probabilities] prefix %s: %s", idx + 1, _format_clbit_order(prefix)
            )
        counts = extract_counts(pub_res)
        filtered_counts, width = _counts_over_measured_clbits(prefix, counts)
        metadata = getattr(pub_res, "metadata", {}) or {}
        declared_total = extract_total_shots(metadata)
        count_total = sum(filtered_counts.values())
        effective_total = _resolve_effective_total_shots(
            count_total,
            declared_total,
            width=width,
        )
        normalized_counts = _normalize_counts_dict(
            filtered_counts,
            width,
            total_shots=effective_total,
        )
        total = sum(normalized_counts.values())
        if total == 0:
            if width == 0 and not normalized_counts:
                probs_list.append({"": 0.0})
            else:
                probs_list.append({key: 0.0 for key in normalized_counts})
            continue
        raw = {bitstr: val / total for bitstr, val in normalized_counts.items()}
        probs_list.append(normalize_probability_dict(raw, num_qubits=width))
    return probs_list


def trace_counts_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    shots: int | None = 4096,
    debug_bit_order: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, int]]:
    """Sample each instruction prefix with a SamplerV2 and return raw counts.

    Args:
        circuit: Circuit to analyze.
        sampler: A SamplerV2-compatible primitive (for example, Aer SamplerV2,
            IBM Runtime SamplerV2, or a statevector-based sampler).
        shots: Number of shots per prefix. When ``None``, the sampler backend's
            configured default is used.
        debug_bit_order: If ``True``, log classical bit order per prefix for debugging.
        parameter_values: Optional parameter bindings applied to each prefix when
            executing the sampler. Accepts the same formats as
            :meth:`QuantumCircuit.assign_parameters` (mapping or sequence). Provide
            a sequence of bindings with one entry per prefix to vary assignments
            between prefixes.

    Returns:
        list[dict[str, int]]: One counts dictionary per prefix using canonical
        bitstring keys consistent with the circuit's classical bit order.

    Raises:
        RuntimeError: If counts cannot be extracted from the sampler result object
            or if the backend does not support mid-circuit measurements.
    """

    if StatevectorSampler is not None and isinstance(sampler, StatevectorSampler):
        if _has_mid_circuit_measurement(circuit):
            raise RuntimeError(_mid_measure_unsupported_message("counts", "StatevectorSampler"))

    prefixes = _prefixes_with_end_measure(circuit)
    base_parameters = tuple(circuit.parameters)
    pubs = _build_sampler_pubs(prefixes, parameter_values, base_parameters)
    normalized_shots = _coerce_optional_non_negative_int(shots, label="shots")

    run_kwargs: Dict[str, Any] = {}
    if normalized_shots is not None:
        run_kwargs["shots"] = normalized_shots

    try:
        job = sampler.run(pubs, **run_kwargs)
        results = job.result()
    except QiskitError as exc:
        _raise_sampler_mid_measure_error("counts", sampler, exc)
        raise  # pragma: no cover - helper always re-raises

    counts_list: List[Dict[str, int]] = []
    for idx, pub_res in enumerate(results):
        prefix = prefixes[idx]
        if debug_bit_order:
            _LOGGER.info("[trace_counts] prefix %s: %s", idx + 1, _format_clbit_order(prefix))
        counts = extract_counts(pub_res)
        filtered_counts, width = _counts_over_measured_clbits(prefix, counts)
        metadata = getattr(pub_res, "metadata", {}) or {}
        declared_total = extract_total_shots(metadata)
        count_total = sum(filtered_counts.values())
        effective_total = _resolve_effective_total_shots(
            count_total,
            declared_total,
            width=width,
        )
        counts_list.append(
            _normalize_counts_dict(
                filtered_counts,
                width,
                total_shots=effective_total,
            )
        )
    return counts_list


def trace_probabilities_with_aer(
    circuit: QuantumCircuit,
    shots: int = 8192,
    method: str = "automatic",
    debug_bit_order: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, float]]:
    """Convenience wrapper that samples using qiskit-aer SamplerV2.

    Requires the optional dependency ``qiskit-aer``.

    Note:
        The ``method`` parameter is currently unused and preserved for possible
        future extensions or compatibility.
    """
    sampler_cls = _load_aer_sampler_cls()
    sampler = sampler_cls()
    return trace_probabilities_with_sampler(
        circuit,
        sampler,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
    )


def trace_counts_with_aer(
    circuit: QuantumCircuit,
    shots: int = 8192,
    method: str = "automatic",
    debug_bit_order: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, int]]:
    """Convenience wrapper that samples counts using qiskit-aer SamplerV2.

    Requires the optional dependency ``qiskit-aer``.

    Note:
        The ``method`` parameter is currently unused and preserved for possible
        future extensions or compatibility.
    """

    sampler_cls = _load_aer_sampler_cls()
    sampler = sampler_cls()
    return trace_counts_with_sampler(
        circuit,
        sampler,
        shots=shots,
        debug_bit_order=debug_bit_order,
        parameter_values=parameter_values,
    )


def trace_marginal_probabilities_with_sampler(
    circuit: QuantumCircuit,
    sampler,
    qubits: List[int],
    shots: int | None = 4096,
    add_measure_for_qubits: bool = False,
    debug_bit_order: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, float]]:
    """Like :func:`trace_probabilities_with_sampler`, but returns marginals.

    Computes probabilities marginalized onto ``qubits`` for each prefix. The
    helper circuits used for sampling always terminate in measurements so that
    counts are available even if the original circuit omitted them. When
    ``add_measure_for_qubits`` is ``False`` (the default), every qubit that has
    not yet been measured in a prefix, or that has been acted upon after its
    most recent measurement, is measured to a scratch classical register. When
    ``True``, only the requested ``qubits`` receive temporary measurements under
    the same rules. If any requested qubit remains unmeasured in a prefix, the
    corresponding marginal entry is an empty dictionary.

    Args:
        circuit: Circuit to analyze.
        sampler: A SamplerV2-compatible primitive.
        qubits: List of qubit indices for the marginal distribution (circuit qubit
            indices, not classical bit positions).
        shots: Number of shots per prefix. When ``None``, the sampler backend's
            configured default is used.
        add_measure_for_qubits: If ``True``, add temporary measurements for
            unmeasured requested qubits to produce the marginals.
        debug_bit_order: If ``True``, log classical bit order and measured mapping.
        parameter_values: Optional parameter bindings applied to each prefix when
            executing the sampler. Accepts the same formats as
            :meth:`QuantumCircuit.assign_parameters` (mapping or sequence). Provide
            a sequence of bindings with one entry per prefix to vary assignments
            between prefixes.

    Returns:
        list[dict[str, float]]: A list of probability dictionaries over the
        requested ``qubits`` per prefix. If none of the requested qubits are
        measured in a prefix and ``add_measure_for_qubits`` is ``False``, the
        corresponding entry is an empty dict.
    """
    # Validate qubit indices early for clear errors
    n_qubits = circuit.num_qubits
    seen_qubits: set[int] = set()
    for q in qubits:
        if not isinstance(q, int) or q < 0 or q >= n_qubits:
            raise ValueError(f"Invalid qubit index {q}; must be 0..{n_qubits-1}.")
        if q in seen_qubits:
            raise ValueError(f"Duplicate qubit index requested: {q}.")
        seen_qubits.add(q)

    if not qubits:
        return [{"": 1.0} for _ in circuit.data]

    if StatevectorSampler is not None and isinstance(sampler, StatevectorSampler):
        if _has_mid_circuit_measurement(circuit):
            raise RuntimeError(_mid_measure_unsupported_message("marginals", "StatevectorSampler"))

    prefixes = (
        _prefixes_with_end_measure_of_qubits(circuit, qubits)
        if add_measure_for_qubits
        else _prefixes_with_end_measure(circuit)
    )
    base_parameters = tuple(circuit.parameters)
    pubs = _build_sampler_pubs(prefixes, parameter_values, base_parameters)
    normalized_shots = _coerce_optional_non_negative_int(shots, label="shots")
    run_kwargs: Dict[str, Any] = {}
    if normalized_shots is not None:
        run_kwargs["shots"] = normalized_shots
    try:
        job = sampler.run(pubs, **run_kwargs)
        results = job.result()
    except QiskitError as exc:
        _raise_sampler_mid_measure_error("marginals", sampler, exc)
        raise  # pragma: no cover - helper always re-raises

    # Build mapping from qubit index -> classical bit index for each prefix
    probs_list: List[Dict[str, float]] = []
    for i, (pref, pub_res) in enumerate(zip(prefixes, results), start=1):
        # Extract counts
        counts = extract_counts(pub_res)

        # Map requested qubits to classical bit positions in this prefix
        measured_map: Dict[int, int] = {}  # qubit -> clbit position (in pref.clbits order)
        for ci in pref.data:
            if ci.operation.name != "measure":
                continue
            if not ci.qubits or not ci.clbits:
                continue
            for qbit, cbit in zip(ci.qubits, ci.clbits):
                q = pref.find_bit(qbit).index
                c = pref.find_bit(cbit).index
                measured_map[q] = c
        if debug_bit_order:
            _LOGGER.info("[trace_marginals] prefix %s: %s", i, _format_clbit_order(pref))
            if measured_map:
                mapping_str = ", ".join(
                    f"q{q}->c{c}:{_label_for_clbit(pref, pref.clbits[c])}"
                    for q, c in sorted(measured_map.items())
                )
                _LOGGER.info("[trace_marginals] measured map: %s", mapping_str)
        keep_clbits = [measured_map[q] for q in qubits if q in measured_map]
        if not keep_clbits:
            probs_list.append({})
            continue
        if len(keep_clbits) != len(qubits):
            if debug_bit_order:
                missing = [q for q in qubits if q not in measured_map]
                _LOGGER.warning(
                    "[trace_marginals] prefix %s missing measurements for qubits: %s",
                    i,
                    missing,
                )
            probs_list.append({})
            continue
        metadata = getattr(pub_res, "metadata", {}) or {}
        declared_total = extract_total_shots(metadata)
        raw_counts = marginalize_counts(counts, keep_clbits)
        count_total = sum(raw_counts.values())
        effective_total = _resolve_effective_total_shots(
            count_total,
            declared_total,
            width=len(keep_clbits),
        )
        normalized_counts = _normalize_counts_dict(
            raw_counts,
            len(keep_clbits),
            total_shots=effective_total,
        )
        total = sum(normalized_counts.values())
        if total == 0:
            probs_list.append({key: 0.0 for key in normalized_counts})
            continue
        probs_list.append({k: v / total for k, v in normalized_counts.items()})
    return probs_list


ObservableOp = Union[Operator, SparsePauliOp]


@dataclass(frozen=True)
class ObsSpec:
    """Normalized observable specification for expectation helpers."""

    name: str
    operator: ObservableOp
    qargs: Optional[Tuple[int, ...]]


def _coerce_observable(op: Union[ObservableOp, Pauli, str, Any]) -> ObservableOp:
    """Return an operator compatible with ``Statevector.expectation_value``.

    ``SparsePauliOp`` inputs are preserved to avoid densifying large observables,
    while other operator-like inputs fall back to the dense
    :class:`~qiskit.quantum_info.Operator` representation.
    """

    if isinstance(op, (Operator, SparsePauliOp)):
        return op
    if isinstance(op, Pauli):
        return SparsePauliOp.from_list([(op.to_label(), 1.0)])
    if isinstance(op, str):
        pauli = Pauli(op)
        return SparsePauliOp.from_list([(pauli.to_label(), 1.0)])
    # Last resort: try to build Operator; may raise if incompatible.
    return Operator(op)


def _normalize_observable_specs(
    circuit: QuantumCircuit,
    observables: Iterable[
        Union[
            ObsSpec,
            Tuple[str, Union[ObservableOp, Pauli, str], Sequence[int]],
            Tuple[str, Union[ObservableOp, Pauli, str]],
            ObservableOp,
            Pauli,
            str,
        ]
    ],
) -> List[ObsSpec]:
    """Return validated observable specifications for ``circuit``.

    The returned list contains tuples of ``(name, Operator, qargs_or_none)`` that are
    guaranteed to be compatible with the circuit width and contain distinct, valid
    qubit indices when ``qargs`` are supplied.
    """

    raw_specs: List[Tuple[str, ObservableOp, Optional[Sequence[int]]]] = []
    auto_idx = 0
    for item in observables:
        if isinstance(item, ObsSpec):
            raw_specs.append((item.name, item.operator, item.qargs))
            continue
        if isinstance(item, tuple):
            if len(item) == 3:
                name, op_like, qargs = item
            elif len(item) == 2:
                name, op_like = cast(Tuple[str, Union[ObservableOp, Pauli, str]], item)
                qargs = None
            else:
                raise ValueError("Observable tuple must be (name, op[, qargs]).")
            raw_specs.append(
                (
                    str(name),
                    _coerce_observable(op_like),
                    tuple(qargs) if qargs is not None else None,
                )
            )
            continue
        op = _coerce_observable(item)
        name = f"obs_{auto_idx}"
        auto_idx += 1
        raw_specs.append((name, op, None))

    n_qubits = circuit.num_qubits

    def _infer_op_qubits(name: str, op: ObservableOp) -> int:
        if isinstance(op, SparsePauliOp):
            return int(op.num_qubits)

        op_qubits = getattr(op, "num_qubits", None)
        if op_qubits is not None:
            return int(op_qubits)

        dim_attr = getattr(op, "dim", None)
        dim: Optional[int] = None
        if dim_attr is not None:
            if isinstance(dim_attr, Sequence):
                if not dim_attr:
                    raise TypeError(f"Cannot infer the number of qubits for observable '{name}'.")
                in_dim = int(dim_attr[0])
                out_dim = int(dim_attr[-1])
                if in_dim != out_dim:
                    raise ValueError(
                        f"Observable '{name}' is not square (dim={dim_attr}); cannot determine qubits."
                    )
                dim = in_dim
            else:
                dim = int(dim_attr)
        else:
            data_attr = getattr(op, "data", None)
            shape = getattr(data_attr, "shape", None)
            if not shape:
                raise TypeError(f"Cannot infer the number of qubits for observable '{name}'.")
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(
                    f"Observable '{name}' has non-square data shape {shape}; cannot determine qubits."
                )
            dim = int(shape[0])

        if dim <= 0:
            raise ValueError(
                f"Observable '{name}' has invalid dimension {dim}; must be a positive power of two."
            )
        if dim & (dim - 1) != 0:
            raise ValueError(
                f"Observable '{name}' has dimension {dim}, which is not a power of two."
            )
        return dim.bit_length() - 1

    specs: List[ObsSpec] = []
    for name, op, qargs in raw_specs:
        op_qubits = _infer_op_qubits(name, op)
        if qargs is None:
            if op_qubits != n_qubits:
                raise ValueError(
                    f"Observable '{name}' acts on {op_qubits} qubits; please provide qargs for a {n_qubits}-qubit circuit."
                )
            specs.append(ObsSpec(name, op, None))
            continue

        seen_qargs: set[int] = set()
        normalized: List[int] = []
        for q in qargs:
            if isinstance(q, bool) or not isinstance(q, Integral):
                raise ValueError(f"Observable '{name}' qargs must be integers; got {q!r}.")
            idx = int(q)
            if idx < 0 or idx >= n_qubits:
                raise ValueError(
                    f"Observable '{name}' qargs contain invalid index {q}; valid range is 0..{n_qubits - 1}."
                )
            if idx in seen_qargs:
                raise ValueError(
                    f"Observable '{name}' qargs contain duplicate index {idx}; each qubit may appear at most once."
                )
            seen_qargs.add(idx)
            normalized.append(idx)
        if len(normalized) != op_qubits:
            raise ValueError(
                f"Observable '{name}' acts on {op_qubits} qubits but qargs has {len(normalized)} entries."
            )
        specs.append(ObsSpec(name, op, tuple(normalized)))

    return specs


@lru_cache(maxsize=None)
def _sparse_identity(num_qubits: int) -> SparsePauliOp:
    """Return a cached sparse identity operator for ``num_qubits`` qubits."""

    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")
    if num_qubits == 0:
        label = ""
    else:
        label = "I" * num_qubits
    return SparsePauliOp.from_list([(label, 1.0)])


def _spec_to_sparse_pauli(spec: ObsSpec, n_qubits: int) -> SparsePauliOp:
    """Return a :class:`~qiskit.quantum_info.SparsePauliOp` acting on ``n_qubits``."""

    op = spec.operator
    if isinstance(op, SparsePauliOp):
        base = op
    else:
        base = SparsePauliOp.from_operator(op)

    qargs = spec.qargs
    name = spec.name

    if qargs is None:
        sparse = base
    else:
        if base.num_qubits != len(qargs):
            raise ValueError(
                f"Observable '{name}' acts on {base.num_qubits} qubits but qargs supplies {len(qargs)} entries."
            )
        identity = _sparse_identity(n_qubits)
        sparse = identity.compose(base, qargs=list(qargs))
    if sparse.coeffs.size:
        max_imag = float(np.max(np.abs(np.imag(sparse.coeffs))))
        if max_imag > 1e-12:
            raise ValueError(
                f"Observable '{name}' produces complex coefficients when expanded for the estimator;"
                " ensure the operator is Hermitian."
            )
    real_coeffs = np.real_if_close(sparse.coeffs, tol=1e-12)
    return SparsePauliOp(sparse.paulis, np.real(real_coeffs))


def _prefix_circuits_without_measurements(
    circuit: QuantumCircuit, include_initial: bool
) -> List[QuantumCircuit]:
    """Return copies of ``circuit`` prefixes without extra measurements."""

    prefixes: List[QuantumCircuit] = []
    if include_initial:
        empty = circuit.copy_empty_like()
        empty.name = "prefix_initial"
        prefixes.append(empty)

    for end in range(1, len(circuit.data) + 1):
        prefix = circuit.copy_empty_like()
        prefix.name = f"prefix_{end - 1}"
        for inst in circuit.data[:end]:
            prefix.append(inst.operation, inst.qubits, inst.clbits)
        prefixes.append(prefix)

    return prefixes


def trace_expectations_with_statevector(
    circuit: QuantumCircuit,
    observables: Iterable[
        Union[
            ObsSpec,
            Tuple[str, Union[ObservableOp, Pauli, str], Sequence[int]],
            Tuple[str, Union[ObservableOp, Pauli, str]],
            ObservableOp,
            Pauli,
            str,
        ]
    ],
    initial_state: Optional[Union[Statevector, int, str, list, tuple]] = None,
    include_initial: bool = False,
    parameter_values: Any = None,
) -> List[Dict[str, float]]:
    """Exact expectation values at each instruction prefix.

    Uses a deterministic, exact statevector stepper so there is no sampling noise.

    Args:
        circuit: Circuit to analyze.
        observables: Iterable of observable specifications. Each item can be one of:
            - ``(name, Operator, qargs)``
            - ``(name, Operator)``
            - ``Operator`` (name auto-assigned)
            - ``SparsePauliOp`` (optionally with ``qargs`` when provided as tuples)
            - ``Pauli`` or one-letter Pauli string ("X", "Y", "Z"). These can also be provided
              within the tuple forms with an explicit name and optional ``qargs``.
            When ``qargs`` is omitted, the operator must act on all qubits of the circuit.
            When provided, ``qargs`` must contain distinct integer qubit indices within the
            circuit range.
        initial_state: Optional initial state for the debugger.
        include_initial: If ``True``, also report expectation values for the initial state
            before any instruction executes.
        parameter_values: Optional parameter bindings applied to the circuit before
            debugging. Accepts the same formats as :meth:`QuantumCircuit.assign_parameters`.

    Returns:
        list[dict[str, float]]: One dict per executed prefix (and optionally the
        initial state) mapping observable name to real-valued expectation.

    Raises:
        ValueError: If observable specifications have inconsistent qubit sizes or
            invalid tuple shapes.
        ValueError: If evaluating an observable yields a complex expectation,
            indicating a non-Hermitian operator.
        TypeError: If the installed Qiskit version does not expose ``qargs`` support
            for expectation values required by one or more observables.
    """
    specs = _normalize_observable_specs(circuit, observables)

    n_qubits = circuit.num_qubits

    dbg = CircuitDebugger(
        circuit,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )
    out: List[Dict[str, float]] = []

    # Optionally compute for initial state before any instruction
    if include_initial:
        vals0: Dict[str, float] = {}
        for spec in specs:
            vals0[spec.name] = _expectation_real(dbg.state, spec.name, spec.operator, spec.qargs)
        out.append(vals0)

    # Step through and record after each executed instruction
    for _ in range(len(circuit.data)):
        rec = dbg.step()
        vals: Dict[str, float] = {}
        for spec in specs:
            vals[spec.name] = _expectation_real(rec.state, spec.name, spec.operator, spec.qargs)
        out.append(vals)
    return out


def trace_expectations_with_estimator(
    circuit: QuantumCircuit,
    observables: Iterable[
        Union[
            ObsSpec,
            Tuple[str, Union[ObservableOp, Pauli, str], Sequence[int]],
            Tuple[str, Union[ObservableOp, Pauli, str]],
            ObservableOp,
            Pauli,
            str,
        ]
    ],
    estimator,
    *,
    include_initial: bool = False,
    parameter_values: Any = None,
    precision: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Estimate expectation values for each prefix using an ``EstimatorV2`` backend.

    Args:
        circuit: Circuit to analyze.
        observables: Iterable of observable specifications. See
            :func:`trace_expectations_with_statevector` for supported formats, including
            ``SparsePauliOp`` inputs.
        estimator: Instance implementing the ``EstimatorV2`` interface (for example
            :class:`qiskit.primitives.StatevectorEstimator` or
            :class:`qiskit.primitives.BackendEstimatorV2`).
        include_initial: If ``True``, also report expectation values for the initial
            state before any instruction executes.
        parameter_values: Optional parameter bindings applied to each prefix when
            executing the estimator. Accepts the same formats as
            :meth:`~qiskit.circuit.QuantumCircuit.assign_parameters`. Provide a
            sequence of bindings with one entry per prefix to vary assignments
            between prefixes. Flat sequences matching the full circuit parameter
            order are reused across prefixes, and dictionary-style bindings are
            filtered per prefix to include only relevant parameters.
        precision: Optional target precision forwarded to the estimator ``run`` call.

    Returns:
        list[dict[str, float]]: One dict per executed prefix (and optionally the
        initial state) mapping observable name to real-valued expectation estimates.

    Raises:
        TypeError: If ``estimator`` does not expose a ``run`` method compatible with
            the EstimatorV2 interface.
        ValueError: If observable specifications are invalid or produce non-Hermitian
            coefficients when expanded to the circuit width.
    """

    if not hasattr(estimator, "run"):
        raise TypeError("estimator must provide a 'run' method compatible with EstimatorV2.")

    for inst in circuit.data:
        if inst.operation.name == "measure":
            raise ValueError(
                "trace_expectations_with_estimator requires circuits without measurements; "
                "please remove measure instructions or use trace_expectations_with_statevector."
            )

    specs = _normalize_observable_specs(circuit, observables)
    n_qubits = circuit.num_qubits

    sparse_ops: List[SparsePauliOp] = []
    names: List[str] = []
    for spec in specs:
        sparse_ops.append(_spec_to_sparse_pauli(spec, n_qubits))
        names.append(spec.name)

    observables_array = ObservablesArray(sparse_ops)
    prefixes = _prefix_circuits_without_measurements(circuit, include_initial)

    if not prefixes:
        return []

    base_parameters = tuple(circuit.parameters)
    pubs = _build_estimator_pubs(prefixes, parameter_values, base_parameters, observables_array)
    run_kwargs: Dict[str, Any] = {}
    if precision is not None:
        run_kwargs["precision"] = precision

    job = estimator.run(pubs, **run_kwargs)
    result = job.result()

    rows: List[Dict[str, float]] = []
    for pub_res in result:
        evs = pub_res.data.evs
        arr = np.asarray(evs)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if np.iscomplexobj(arr):
            imag = np.abs(np.imag(arr))
            max_imag = float(np.max(imag)) if arr.size else 0.0
            real = np.abs(np.real(arr))
            max_real = float(np.max(real)) if arr.size else 0.0
            tolerance = 1e-12 * max(1.0, max_real)
            if max_imag > tolerance:
                raise ValueError(
                    "Estimator returned complex expectation values; ensure all observables are Hermitian."
                )
            arr = np.real(arr)
        else:
            arr = arr.astype(float, copy=False)
        try:
            reshaped = arr.reshape(-1, len(names))
        except ValueError as exc:
            raise ValueError(
                "Estimator returned expectation data with unexpected size; "
                f"expected a multiple of {len(names)} values per prefix but received {arr.size}."
            ) from exc
        means = reshaped.mean(axis=0)
        row = {name: float(means[idx]) for idx, name in enumerate(names)}
        rows.append(row)

    return rows


@lru_cache(maxsize=None)
def _callable_accepts_qargs(func) -> bool:
    """Return ``True`` if ``func`` accepts a ``qargs`` keyword argument."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - fallback to runtime check
        return True

    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "qargs" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _qargs_type_error(exc: TypeError) -> bool:
    """Return ``True`` when ``exc`` indicates that ``qargs`` is unsupported."""

    text = " ".join(str(arg) for arg in exc.args).lower()
    return "qarg" in text


def _expectation_real(
    state: Statevector, name: str, op: ObservableOp, qargs: Optional[Tuple[int, ...]]
) -> float:
    """Return a real expectation value or raise if the result is complex."""

    method = state.expectation_value
    if qargs is None:
        ev = method(op)
    else:
        target = getattr(method, "__func__", method)
        if not _callable_accepts_qargs(target):
            raise TypeError(
                "Statevector.expectation_value does not support the 'qargs' keyword "
                f"required for observable '{name}' with qargs={qargs}. "
                "Qiskit Inspect requires qiskit>=2.0.0, where this keyword is available."
            )
        try:
            ev = method(op, qargs=list(qargs))
        except TypeError as exc:
            if _qargs_type_error(exc):
                raise TypeError(
                    "Statevector.expectation_value does not support the 'qargs' keyword "
                    f"required for observable '{name}' with qargs={qargs}. "
                    "Qiskit Inspect requires qiskit>=2.0.0, where this keyword is available."
                ) from exc
            raise
    if abs(ev.imag) > 1e-12:
        raise ValueError(
            f"Observable '{name}' produced a complex expectation value {ev}; ensure the operator is Hermitian."
        )
    return float(ev.real)


def trace_probabilities_with_statevector_exact(
    circuit: QuantumCircuit,
    include_initial: bool = False,
    initial_state: Optional[Union[Statevector, int, str, list, tuple]] = None,
    parameter_values: Any = None,
    *,
    flatten_control_flow: bool = False,
) -> List[Dict[str, float]]:
    """Exact computational-basis probabilities after each prefix.

    Uses the exact statevector stepper (no sampling). Keys follow
    :meth:`qiskit.quantum_info.Statevector.probabilities_dict` conventions.

    Args:
        circuit: Circuit to analyze.
        include_initial: If ``True``, include probabilities of the initial state
            prior to any instruction.
        initial_state: Optional initial quantum state for the simulator. Accepts
            :class:`~qiskit.quantum_info.Statevector`, integers, bitstrings, or
            amplitude sequences compatible with :class:`Statevector`.
        parameter_values: Optional parameter bindings applied before tracing the
            circuit. Accepts the same formats as :meth:`QuantumCircuit.assign_parameters`.
        flatten_control_flow: When ``True``, record the state after every executed
            instruction inside nested control-flow blocks (for example, each
            iteration of a :class:`~qiskit.circuit.controlflow.ForLoopOp`).

    Returns:
        list[dict[str, float]]: Probabilities per prefix (and optionally initial).
    """
    if flatten_control_flow:
        raise ValueError(
            "trace_probabilities_with_statevector_exact does not support flatten_control_flow=True; "
            "please leave control flow unflattened to preserve execution structure."
        )

    dbg = CircuitDebugger(
        circuit,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )

    total_steps = len(circuit.data)
    accum: List[Dict[str, float]] = [defaultdict(float) for _ in range(total_steps)]

    def _record(step_index: int, probs: Dict[str, float], weight: float) -> None:
        bucket = accum[step_index]
        for key, value in probs.items():
            bucket[key] += weight * value

    def _project_state(state: Statevector, qubit_indices: List[int], outcome: Tuple[int, ...]):
        data = np.asarray(state.data, dtype=complex)
        if not qubit_indices:
            return state.copy(), 1.0
        mask = 0
        target = 0
        for qi, bit in zip(qubit_indices, outcome):
            mask |= 1 << qi
            if bit:
                target |= 1 << qi
        indices = np.arange(data.size)
        selector = (indices & mask) == target
        selected = data[selector]
        prob = float(np.sum(np.abs(selected) ** 2))
        if prob == 0.0:
            return None, 0.0
        collapsed = np.zeros_like(data)
        collapsed[selector] = selected / np.sqrt(prob)
        return Statevector(collapsed, dims=state.dims()), prob

    def _explore(dbg_branch: CircuitDebugger, weight: float) -> None:
        while dbg_branch._ip < len(dbg_branch.instructions):
            inst_context = dbg_branch.instructions[dbg_branch._ip]
            inst = inst_context.operation
            if inst.name == "measure" and inst_context.qubits:
                cond = getattr(inst, "condition", None)
                if cond is not None:
                    if isinstance(cond, tuple):
                        should_apply = dbg_branch._eval_condition(cond)
                    else:
                        cond_eval = getattr(dbg_branch, "_cond_eval", None)
                        if cond_eval is not None:
                            try:
                                should_apply = bool(
                                    cond_eval(cond, dbg_branch.classical_bits, dbg_branch.circuit)
                                )
                            except Exception:
                                should_apply = dbg_branch._eval_condition_object(cond)
                        else:
                            should_apply = dbg_branch._eval_condition_object(cond)
                    if not should_apply:
                        step_slot = dbg_branch._ip
                        dbg_branch._ip += 1
                        _record(
                            step_slot, _probabilities_from_statevector(dbg_branch.state), weight
                        )
                        continue
                q_indices = [dbg_branch.circuit.find_bit(q).index for q in inst_context.qubits]
                c_indices = [dbg_branch.circuit.find_bit(c).index for c in inst_context.clbits]
                for outcome in itertools.product((0, 1), repeat=len(q_indices)):
                    projected, prob = _project_state(dbg_branch.state, q_indices, outcome)
                    if projected is None or prob == 0.0:
                        continue
                    child = copy.deepcopy(dbg_branch)
                    child.state = projected
                    for qi, bit in zip(c_indices, outcome):
                        child.classical_bits[qi] = int(bit)
                    if len(c_indices) > len(outcome):
                        for extra in c_indices[len(outcome) :]:
                            child.classical_bits[extra] = 0
                    child._ip += 1
                    collapsed_probs = _probabilities_from_statevector(projected)
                    _record(child._ip - 1, collapsed_probs, weight * prob)
                    _explore(child, weight * prob)
                return
            rec = dbg_branch.step()
            _record(rec.step_index - 1, _probabilities_from_statevector(rec.state), weight)

    results: List[Dict[str, float]] = []
    if include_initial:
        results.append(_probabilities_from_statevector(dbg.state))
    _explore(copy.deepcopy(dbg), 1.0)
    results.extend(dict(sorted(bucket.items())) for bucket in accum)
    return results


def trace_statevectors_with_statevector_exact(
    circuit: QuantumCircuit,
    include_initial: bool = False,
    initial_state: Optional[Union[Statevector, int, str, list, tuple]] = None,
    parameter_values: Any = None,
    *,
    flatten_control_flow: bool = False,
) -> List[Statevector]:
    """Return the exact statevector after each instruction prefix.

    This is a thin convenience wrapper around :class:`CircuitDebugger` that
    collects the intermediate :class:`~qiskit.quantum_info.Statevector`
    snapshots produced while stepping through ``circuit``.  Qiskit itself does
    not expose an API to retrieve per-instruction statevectors, so callers
    previously had to instantiate :class:`CircuitDebugger` directly.  The helper
    mirrors :func:`trace_probabilities_with_statevector_exact` but returns raw
    statevectors instead of probability dictionaries.

    Args:
        circuit: Circuit to analyze deterministically.
        include_initial: If ``True``, include the initial state prior to any
            instruction executing.
        initial_state: Optional starting state for the debugger.  Accepts the
            same formats as :class:`Statevector` (integer basis label, bitstring
            label, iterable of amplitudes, or an existing statevector).
        parameter_values: Optional parameter bindings applied before tracing the
            circuit.  Accepts the same formats as
            :meth:`QuantumCircuit.assign_parameters`.
        flatten_control_flow: When ``True``, include intermediate statevectors for
            operations executed inside control-flow blocks.

    Returns:
        list[Statevector]: Copies of the statevector after each executed
        instruction (and optionally the initial state).
    """

    dbg = CircuitDebugger(
        circuit,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )

    records = dbg.trace(
        include_initial=include_initial,
        flatten_control_flow=flatten_control_flow,
    )
    return [rec.state.copy() for rec in records]


def trace_marginal_probabilities_with_statevector(
    circuit: QuantumCircuit,
    qubits: List[int],
    include_initial: bool = False,
    initial_state: Optional[Union[Statevector, int, str, list, tuple]] = None,
    parameter_values: Any = None,
    *,
    flatten_control_flow: bool = False,
) -> List[Dict[str, float]]:
    """Exact marginal probabilities over selected qubits at each prefix.

    The bitstring keys are ordered according to the provided ``qubits`` list
    (the first element is the leftmost bit in the key). If ``include_initial`` is
    ``True``, the first entry corresponds to the initial state.

    Args:
        circuit: Circuit to analyze.
        qubits: Circuit qubit indices to keep in the marginal distribution.
        include_initial: If ``True``, include the initial state's marginals.
        initial_state: Optional initial state specification passed through to
            :class:`CircuitDebugger`.
        parameter_values: Optional parameter bindings applied before tracing the
            circuit. Accepts the same formats as :meth:`QuantumCircuit.assign_parameters`.
        flatten_control_flow: When ``True``, record the marginal distribution after
            every instruction executed inside nested control-flow blocks.

    Returns:
        list[dict[str, float]]: Marginal probabilities per prefix (and optionally initial).

    Raises:
        ValueError: If ``qubits`` contains duplicates or indices outside the circuit range.
    """

    n_qubits = circuit.num_qubits
    seen: set[int] = set()
    for q in qubits:
        if not isinstance(q, int) or q < 0 or q >= n_qubits:
            raise ValueError(f"Invalid qubit index {q}; must be 0..{n_qubits - 1}.")
        if q in seen:
            raise ValueError(f"Duplicate qubit index requested: {q}.")
        seen.add(q)

    def _marginal_probs_dict(state: Statevector) -> Dict[str, float]:
        if not qubits:
            return {"": 1.0}

        # Use probabilities(qargs=...) to get values, and format keys so that
        # the leftmost bit corresponds to the first qubit in the provided `qubits` list.
        arr = state.probabilities(qargs=qubits)
        k = len(qubits)
        out: Dict[str, float] = {}
        for i, v in enumerate(arr):
            val = float(v)
            if val <= 0.0:
                continue
            bits = format(i, f"0{k}b")
            rounded = round(val, 15)
            out[bits] = rounded if rounded != 0.0 or val == 0.0 else val
        # If all entries were numerically zero (or the statevector probabilities are
        # extremely small), keep the largest value to preserve normalization.
        if not out and len(arr) > 0:
            imax = max(range(len(arr)), key=lambda j: float(arr[j]))
            val = float(arr[imax])
            bits = format(imax, f"0{k}b")
            rounded = round(val, 15)
            out[bits] = rounded if rounded != 0.0 or val == 0.0 else val
        return out

    dbg = CircuitDebugger(
        circuit,
        initial_state=initial_state,
        parameter_values=parameter_values,
    )

    records = dbg.trace(
        include_initial=include_initial,
        flatten_control_flow=flatten_control_flow,
    )
    return [_marginal_probs_dict(rec.state) for rec in records]


def _raise_sampler_mid_measure_error(helper: _MidMeasureHelper, sampler, exc: QiskitError) -> None:
    """Re-raise ``exc`` with a clearer explanation for unsupported samplers."""

    message = " ".join(str(arg) for arg in exc.args)
    normalized = message.replace("-", " ").lower()
    if "mid circuit" not in normalized:
        raise exc
    backend_label = (
        "StatevectorSampler"
        if StatevectorSampler is not None and isinstance(sampler, StatevectorSampler)
        else "the provided sampler backend"
    )
    raise RuntimeError(_mid_measure_unsupported_message(helper, backend_label)) from exc
