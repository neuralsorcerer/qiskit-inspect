from __future__ import annotations

import importlib
import importlib.util
import math

import pytest
from qiskit import QuantumCircuit

from qiskit_inspect import trace_probabilities_with_aer, trace_probabilities_with_statevector_exact

# Require Aer SamplerV2 and Qiskit Terra >= 2.0 to avoid deprecated APIs
sampler_v2_spec = importlib.util.find_spec("qiskit_aer.primitives")
terra_ok = False
try:
    terra_mod = importlib.import_module("qiskit")
    from packaging.version import Version

    terra_ok = Version(getattr(terra_mod, "__version__", "0")) >= Version("2.0.0")
except Exception:
    terra_ok = False

requires_aer = pytest.mark.skipif(
    sampler_v2_spec is None or not terra_ok,
    reason="Requires qiskit-aer SamplerV2 and Qiskit Terra >= 2.0",
)


def l1_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p.keys()).union(q.keys())
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@requires_aer
@pytest.mark.parametrize("shots", [2048, 4096, 8192])
def test_aer_matches_exact_within_tolerance(shots: int):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    exact = trace_probabilities_with_statevector_exact(qc, include_initial=False)[-1]
    aer = trace_probabilities_with_aer(qc, shots=shots, method="automatic")[-1]

    # l1 should shrink with more shots; we allow a tolerance that scales like ~ 1/sqrt(shots)
    tol = 3.5 / math.sqrt(shots)
    assert l1_distance(exact, aer) < tol
