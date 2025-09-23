# Enforce supported Qiskit runtime requirements
from . import _version_guard

ensure_supported_qiskit_version = _version_guard.ensure_supported_qiskit_version

# re-export public API
from .analytics import (
    cross_entropy,
    hellinger_distance,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    prefix_cross_entropies,
    prefix_hellinger_distances,
    prefix_jensen_shannon_divergences,
    prefix_kullback_leibler_divergences,
    prefix_shannon_entropies,
    prefix_total_variation_distances,
    shannon_entropy,
    total_variation_distance,
    trace_shannon_entropy_with_sampler,
    trace_shannon_entropy_with_statevector,
)
from .assertions import assert_counts_close, assert_probabilities_close, assert_state_equiv
from .backend_trace import (
    trace_counts_with_aer,
    trace_counts_with_sampler,
    trace_expectations_with_estimator,
    trace_expectations_with_statevector,
    trace_marginal_probabilities_with_sampler,
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_aer,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
    trace_statevectors_with_statevector_exact,
)
from .debugger import CircuitDebugger, TraceRecord
from .export import (
    counts_to_dataframe,
    expectations_to_dataframe,
    probabilities_to_dataframe,
    trace_records_to_dataframe,
    write_expectations_csv,
    write_expectations_json,
    write_trace_csv,
    write_trace_json,
)
from .trace_logging import enable_trace_logging
from .visual import ascii_histogram, format_classical_bits, pretty_ket, top_amplitudes

__all__ = [
    "CircuitDebugger",
    "TraceRecord",
    "trace_counts_with_sampler",
    "trace_counts_with_aer",
    "trace_probabilities_with_sampler",
    "trace_probabilities_with_aer",
    "trace_probabilities_with_statevector_exact",
    "trace_statevectors_with_statevector_exact",
    "trace_marginal_probabilities_with_sampler",
    "trace_marginal_probabilities_with_statevector",
    "trace_shannon_entropy_with_statevector",
    "trace_shannon_entropy_with_sampler",
    "trace_expectations_with_estimator",
    "trace_expectations_with_statevector",
    "assert_state_equiv",
    "assert_probabilities_close",
    "assert_counts_close",
    "shannon_entropy",
    "prefix_shannon_entropies",
    "prefix_total_variation_distances",
    "prefix_cross_entropies",
    "prefix_kullback_leibler_divergences",
    "prefix_jensen_shannon_divergences",
    "prefix_hellinger_distances",
    "total_variation_distance",
    "cross_entropy",
    "kullback_leibler_divergence",
    "jensen_shannon_divergence",
    "hellinger_distance",
    "pretty_ket",
    "top_amplitudes",
    "ascii_histogram",
    "format_classical_bits",
    "write_expectations_csv",
    "write_expectations_json",
    "write_trace_csv",
    "write_trace_json",
    "trace_records_to_dataframe",
    "probabilities_to_dataframe",
    "counts_to_dataframe",
    "expectations_to_dataframe",
    "enable_trace_logging",
    "ensure_supported_qiskit_version",
]
