# Ensure the repository root is importable during tests (defensive for various runners).
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - optional primitive depending on qiskit build
    from qiskit.primitives import StatevectorSampler  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - older qiskit builds
    StatevectorSampler = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def statevector_sampler():
    """Provide a ``StatevectorSampler`` instance when available."""

    if StatevectorSampler is None:
        pytest.skip("StatevectorSampler is unavailable in this environment.")
    return StatevectorSampler()
