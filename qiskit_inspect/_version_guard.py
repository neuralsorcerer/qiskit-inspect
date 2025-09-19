"""Runtime enforcement of the supported Qiskit version."""

from __future__ import annotations

import re
from importlib import import_module
from types import ModuleType
from typing import Tuple

_MINIMUM_QISKIT_VERSION = "2.0.0"

try:  # pragma: no cover - import succeeds when qiskit is installed
    from qiskit import __version__ as _INSTALLED_QISKIT_VERSION  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
    raise ImportError(
        f"Qiskit Inspect requires qiskit>={_MINIMUM_QISKIT_VERSION}; "
        "install qiskit 2.0 or newer to continue."
    ) from exc

try:  # pragma: no cover - packaging is optional at runtime
    _packaging_version: ModuleType | None = import_module("packaging.version")
except Exception:  # pragma: no cover - fall back to manual parsing
    _packaging_version = None

if _packaging_version is not None:  # pragma: no branch - import succeeded
    _VersionClass = getattr(_packaging_version, "Version", None)
    _InvalidVersionClass = getattr(_packaging_version, "InvalidVersion", Exception)
else:  # pragma: no cover - packaging not available
    _VersionClass = None
    _InvalidVersionClass = Exception

__all__ = ["ensure_supported_qiskit_version"]
_VERSION_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _collect_missing_features() -> list[str]:
    """Return a list of required Qiskit features that are unavailable."""

    missing: list[str] = []

    def _record_missing(module_name: str, attribute: str | None = None) -> None:
        if attribute is None:
            missing.append(module_name)
        else:
            missing.append(f"{module_name}.{attribute}")

    try:
        primitives = import_module("qiskit.primitives")
    except Exception:
        _record_missing("qiskit.primitives")
    else:
        if not hasattr(primitives, "StatevectorSampler"):
            _record_missing("qiskit.primitives", "StatevectorSampler")
        if not hasattr(primitives, "StatevectorEstimator"):
            _record_missing("qiskit.primitives", "StatevectorEstimator")
        try:
            containers = import_module("qiskit.primitives.containers")
        except Exception:
            _record_missing("qiskit.primitives", "containers")
        else:
            for attribute in ("BitArray", "DataBin"):
                if not hasattr(containers, attribute):
                    _record_missing("qiskit.primitives.containers", attribute)
        try:
            observables_mod = import_module("qiskit.primitives.containers.observables_array")
        except Exception:
            _record_missing("qiskit.primitives.containers", "observables_array.ObservablesArray")
        else:
            if not hasattr(observables_mod, "ObservablesArray"):
                _record_missing(
                    "qiskit.primitives.containers", "observables_array.ObservablesArray"
                )

    try:
        classical = import_module("qiskit.circuit.classical")
    except Exception:
        _record_missing("qiskit.circuit", "classical")
    else:
        if not hasattr(classical, "expr"):
            _record_missing("qiskit.circuit.classical", "expr")
        else:
            expr_module = getattr(classical, "expr")
            if not hasattr(expr_module, "Expr") or not hasattr(expr_module, "ExprVisitor"):
                _record_missing("qiskit.circuit.classical.expr", "ExprVisitor")
        if not hasattr(classical, "types"):
            _record_missing("qiskit.circuit.classical", "types")

    try:
        controlflow = import_module("qiskit.circuit.controlflow")
    except Exception:
        _record_missing("qiskit.circuit", "controlflow")
    else:
        for name in (
            "ForLoopOp",
            "WhileLoopOp",
            "SwitchCaseOp",
            "BreakLoopOp",
            "ContinueLoopOp",
        ):
            if not hasattr(controlflow, name):
                _record_missing("qiskit.circuit.controlflow", name)

    try:
        circuit = import_module("qiskit.circuit")
    except Exception:
        _record_missing("qiskit.circuit")
    else:
        if not hasattr(circuit, "IfElseOp"):
            _record_missing("qiskit.circuit", "IfElseOp")

    return missing


def _parse_version_tuple(text: str) -> Tuple[int, int, int]:
    """Return ``text`` as a ``(major, minor, patch)`` tuple.

    The parser understands semantic-version-like strings. Pre-release and build
    metadata suffixes are ignored so long as the core version components start
    with integers (for example, ``"2.0.0.dev1"`` -> ``(2, 0, 0)``). If the
    string cannot be interpreted, :class:`ValueError` is raised.
    """

    match = _VERSION_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Unable to interpret version string '{text}'.")
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    return major, minor, patch


def ensure_supported_qiskit_version(
    minimum: str = _MINIMUM_QISKIT_VERSION,
    installed_version: str | None = None,
) -> None:
    """Raise if the installed Qiskit version is older than ``minimum``.

    Args:
        minimum: Minimum supported version expressed as a string. Defaults to
            ``"2.0.0"``.
        installed_version: Override for the version string to validate. Primarily
            intended for tests. When omitted, the import-time Qiskit
            ``__version__`` is used.

    Raises:
        ImportError: If the installed version cannot be interpreted or is older
            than ``minimum``.
    """

    target_version = installed_version or _INSTALLED_QISKIT_VERSION
    normalized = target_version.strip()
    if not normalized:
        raise ImportError(
            f"Qiskit Inspect requires qiskit>={minimum}; the installed version string is empty."
        )

    if _VersionClass is not None:  # pragma: no branch - exercised under normal installs
        try:
            parsed_installed = _VersionClass(normalized)
            parsed_required = _VersionClass(minimum)
        except _InvalidVersionClass:
            pass
        else:
            if parsed_installed < parsed_required:
                installed_release = parsed_installed.release
                required_release = parsed_required.release
                if installed_release and required_release and installed_release >= required_release:
                    # Allow pre-release or development builds when their release
                    # tuple meets or exceeds the required release (e.g. 2.0.0rc1
                    # satisfying a 2.0.0 floor).
                    missing_features = _collect_missing_features()
                    if missing_features:
                        formatted = ", ".join(sorted(missing_features))
                        raise ImportError(
                            "Qiskit Inspect requires qiskit>=2.0.0 with full control-flow and primitives "
                            f"support; the following components are unavailable: {formatted}."
                        )
                    return
                raise ImportError(
                    f"Qiskit Inspect requires qiskit>={minimum}; found {target_version}."
                )
            missing_features = _collect_missing_features()
            if missing_features:
                formatted = ", ".join(sorted(missing_features))
                raise ImportError(
                    "Qiskit Inspect requires qiskit>=2.0.0 with full control-flow and primitives "
                    f"support; the following components are unavailable: {formatted}."
                )
            return

    try:
        installed_tuple = _parse_version_tuple(normalized)
        required_tuple = _parse_version_tuple(minimum)
    except ValueError as exc:
        raise ImportError(
            f"Qiskit Inspect requires qiskit>={minimum}; "
            f"unable to interpret installed version '{target_version}'."
        ) from exc

    if installed_tuple < required_tuple:
        raise ImportError(f"Qiskit Inspect requires qiskit>={minimum}; found {target_version}.")

    missing_features = _collect_missing_features()
    if missing_features:
        formatted = ", ".join(sorted(missing_features))
        raise ImportError(
            "Qiskit Inspect requires qiskit>=2.0.0 with full control-flow and primitives "
            f"support; the following components are unavailable: {formatted}."
        )


# Validate the active environment when the package is imported.
ensure_supported_qiskit_version()
