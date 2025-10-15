import math

import pytest
from qiskit import QuantumCircuit

from qiskit_inspect.analytics import (
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


@pytest.mark.parametrize(
    "distribution, expected",
    [
        ({"0": 1.0}, 0.0),
        ({"0": 0.5, "1": 0.5}, 1.0),
        ({"00": 0.25, "11": 0.75}, -(0.25 * math.log(0.25, 2) + 0.75 * math.log(0.75, 2))),
    ],
)
def test_shannon_entropy_probabilities(distribution, expected):
    entropy = shannon_entropy(distribution)
    assert math.isclose(entropy, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_shannon_entropy_counts_handles_integers():
    counts = {"0": 3, "1": 1}
    entropy = shannon_entropy(counts)
    expected = -(0.75 * math.log(0.75, 2) + 0.25 * math.log(0.25, 2))
    assert math.isclose(entropy, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_prefix_shannon_entropies_accepts_width_sequence():
    prefixes = [{"0": 1.0}, {"00": 0.5, "11": 0.5}]
    entropies = prefix_shannon_entropies(prefixes, num_qubits=[1, 2])
    assert entropies[0] == 0.0
    assert math.isclose(entropies[1], 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_shannon_entropy_rejects_invalid_base():
    with pytest.raises(ValueError):
        shannon_entropy({"0": 1.0}, base=1.0)


def test_shannon_entropy_rejects_nonfinite_base():
    with pytest.raises(ValueError):
        shannon_entropy({"0": 1.0}, base=float("nan"))


def test_total_variation_distance_from_counts():
    first = {"0": 30, "1": 10}
    second = {"0": 20, "1": 20}
    distance = total_variation_distance(first, second)
    assert math.isclose(distance, 0.25, rel_tol=0, abs_tol=1e-12)


def test_total_variation_distance_rejects_width_mismatch():
    with pytest.raises(ValueError):
        total_variation_distance({"0": 1.0}, {"00": 1.0})


def test_trace_shannon_entropy_with_statevector_includes_initial():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    entropies = trace_shannon_entropy_with_statevector(qc, include_initial=True)
    assert len(entropies) == len(qc.data) + 1
    assert entropies[0] == 0.0
    assert entropies[-1] == 1.0


@pytest.mark.filterwarnings("ignore:Requested shots not used by sampler backend")
def test_trace_shannon_entropy_with_sampler_statevector_sampler(statevector_sampler):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    entropies = trace_shannon_entropy_with_sampler(qc, statevector_sampler, shots=1024)
    assert len(entropies) == len(qc.data)
    assert 0.9 <= entropies[-1] <= 1.0


def test_total_variation_distance_matches_manual():
    p = {"0": 0.5, "1": 0.5}
    q = {"0": 1.0}
    distance = total_variation_distance(p, q)
    assert math.isclose(distance, 0.5, rel_tol=0, abs_tol=1e-12)


def test_prefix_shannon_entropies_rejects_negative_width():
    with pytest.raises(ValueError):
        prefix_shannon_entropies([{0: 1.0}], num_qubits=[-1])


def test_shannon_entropy_rejects_non_integral_num_qubits():
    with pytest.raises(TypeError):
        shannon_entropy({"0": 1.0}, num_qubits=1.5)


def test_cross_entropy_matches_manual():
    p = {"0": 0.25, "1": 0.75}
    q = {"0": 0.5, "1": 0.5}
    result = cross_entropy(p, q)
    expected = -(0.25 * math.log(0.5, 2) + 0.75 * math.log(0.5, 2))
    assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-12)


def test_cross_entropy_rejects_missing_support():
    with pytest.raises(ValueError):
        cross_entropy({"0": 1.0}, {"1": 1.0})


def test_cross_entropy_rejects_invalid_base():
    with pytest.raises(ValueError):
        cross_entropy({"0": 1.0}, {"0": 1.0}, base=1.0)


def test_kullback_leibler_divergence_matches_manual():
    p = {"0": 0.25, "1": 0.75}
    q = {"0": 0.5, "1": 0.5}
    result = kullback_leibler_divergence(p, q)
    expected = 0.25 * math.log(0.25 / 0.5, 2) + 0.75 * math.log(0.75 / 0.5, 2)
    assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-12)


def test_kl_divergence_rejects_missing_support():
    with pytest.raises(ValueError):
        kullback_leibler_divergence({"0": 1.0}, {"1": 1.0})


def test_jensen_shannon_divergence_is_symmetric_and_bounded():
    p = {"0": 0.6, "1": 0.4}
    q = {"0": 0.1, "1": 0.9}
    forward = jensen_shannon_divergence(p, q)
    reverse = jensen_shannon_divergence(q, p)
    assert math.isclose(forward, reverse, rel_tol=0, abs_tol=1e-12)
    assert 0.0 <= forward <= 1.0


def test_hellinger_distance_matches_manual():
    p = {"0": 0.25, "1": 0.75}
    q = {"0": 0.5, "1": 0.5}
    result = hellinger_distance(p, q)
    expected = math.sqrt(
        0.5 * ((math.sqrt(0.25) - math.sqrt(0.5)) ** 2 + (math.sqrt(0.75) - math.sqrt(0.5)) ** 2)
    )
    assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-12)


def test_prefix_total_variation_distances_against_reference():
    prefixes = [{"0": 1.0}, {"0": 0.5, "1": 0.5}]
    reference = {"0": 1.0}
    distances = prefix_total_variation_distances(prefixes, reference)
    assert distances[0] == 0.0
    assert math.isclose(distances[1], 0.5, rel_tol=0, abs_tol=1e-12)


def test_prefix_metric_sequence_supports_per_prefix_reference():
    prefixes = [{"0": 1.0}, {"0": 0.5, "1": 0.5}]
    reference = [{"0": 1.0}, {"0": 0.75, "1": 0.25}]
    distances = prefix_total_variation_distances(prefixes, reference)
    assert distances[0] == 0.0
    expected = 0.5 * (abs(0.5 - 0.75) + abs(0.5 - 0.25))
    assert math.isclose(distances[1], expected, rel_tol=0, abs_tol=1e-12)


def test_prefix_metric_sequence_allows_nan_entries():
    prefixes = [{"0": 1.0}, {"0": 0.5, "1": 0.5}]
    reference = [{"0": 1.0}, None]
    distances = prefix_total_variation_distances(prefixes, reference)
    assert distances[0] == 0.0
    assert math.isnan(distances[1])


def test_prefix_metric_sequence_requires_reference_length():
    prefixes = [{"0": 1.0}, {"0": 0.5, "1": 0.5}]
    with pytest.raises(ValueError):
        prefix_total_variation_distances(prefixes, [{"0": 1.0}])


def test_prefix_cross_entropies_matches_scalar():
    prefixes = [{"0": 1.0}, {"0": 0.25, "1": 0.75}]
    reference = {"0": 0.5, "1": 0.5}
    series = prefix_cross_entropies(prefixes, reference)
    assert series[0] == cross_entropy(prefixes[0], reference)
    assert math.isclose(series[1], cross_entropy(prefixes[1], reference), rel_tol=0, abs_tol=1e-12)


def test_prefix_kl_divergences_match_scalar():
    prefixes = [{"0": 1.0}, {"0": 0.25, "1": 0.75}]
    reference = {"0": 0.5, "1": 0.5}
    series = prefix_kullback_leibler_divergences(prefixes, reference)
    assert series[0] == kullback_leibler_divergence(prefixes[0], reference)
    assert math.isclose(
        series[1],
        kullback_leibler_divergence(prefixes[1], reference),
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_prefix_metrics_forward_errors_with_index():
    prefixes = [{"0": 1.0}, {"0": 0.5, "1": 0.5}]
    reference = [{"1": 1.0}, {"0": 0.5, "1": 0.5}]
    with pytest.raises(ValueError) as exc:
        prefix_cross_entropies(prefixes, reference)
    assert "prefix index 0" in str(exc.value)


def test_prefix_js_and_hellinger_match_scalar():
    prefixes = [{"0": 1.0}, {"0": 0.25, "1": 0.75}]
    reference = {"0": 0.5, "1": 0.5}
    js = prefix_jensen_shannon_divergences(prefixes, reference)
    h = prefix_hellinger_distances(prefixes, reference)
    assert js[0] == jensen_shannon_divergence(prefixes[0], reference)
    assert h[0] == hellinger_distance(prefixes[0], reference)
    assert math.isclose(
        js[1], jensen_shannon_divergence(prefixes[1], reference), rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(h[1], hellinger_distance(prefixes[1], reference), rel_tol=0, abs_tol=1e-12)
