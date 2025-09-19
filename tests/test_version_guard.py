"""Tests for the Qiskit version guard."""

from __future__ import annotations

import builtins
import importlib

import pytest

import qiskit_inspect._version_guard as version_guard


@pytest.mark.parametrize(
    "installed",
    [
        "2.0.0",
        "2.1.2",
        "2.0.0.dev1",
        "2.0",
        "2.0.0rc3",
        "2.1.0rc1",
    ],
)
def test_version_guard_accepts_supported_versions(installed):
    version_guard.ensure_supported_qiskit_version(installed_version=installed)


@pytest.mark.parametrize(
    "installed",
    [
        "0.45.3",
        "1.2.0",
        "1.99.99",
    ],
)
def test_version_guard_rejects_legacy_versions(installed):
    with pytest.raises(ImportError, match="requires qiskit>=2.0.0"):
        version_guard.ensure_supported_qiskit_version(installed_version=installed)


def test_version_guard_rejects_prerelease_below_required():
    with pytest.raises(ImportError, match="requires qiskit>=2.1.0"):
        version_guard.ensure_supported_qiskit_version(minimum="2.1.0", installed_version="2.0.0rc1")


def test_version_guard_rejects_unparseable_versions():
    module = importlib.reload(version_guard)
    try:
        with pytest.raises(ImportError, match="unable to interpret installed version 'main'"):
            module.ensure_supported_qiskit_version(installed_version="main")
    finally:
        importlib.reload(version_guard)


def test_version_guard_missing_qiskit(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        original_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "qiskit" or name.startswith("qiskit."):
                raise ModuleNotFoundError("No module named 'qiskit'")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(
            ImportError, match="requires qiskit>=2.0.0; install qiskit 2.0 or newer"
        ):
            importlib.reload(module)
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import, raising=False)
        importlib.reload(version_guard)


def test_version_guard_manual_parsing(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        monkeypatch.setattr(module, "_VersionClass", None)
        module.ensure_supported_qiskit_version(installed_version="2.0.0")
        module.ensure_supported_qiskit_version(installed_version="2.1.0rc1", minimum="2.1.0")
        with pytest.raises(ImportError, match="requires qiskit>=2.1.0"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0rc1", minimum="2.1.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_statevector_sampler(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives":
                raise ImportError("missing primitives")
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(ImportError, match="qiskit.primitives"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_statevector_sampler_attribute(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives":
                from types import SimpleNamespace

                return SimpleNamespace()
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(ImportError, match="StatevectorSampler"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_classical_expr(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.circuit.classical":
                from types import SimpleNamespace

                return SimpleNamespace()
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(ImportError, match="classical"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_primitive_containers(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives.containers":
                raise ImportError("missing containers")
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(ImportError, match="qiskit.primitives\\.containers"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


@pytest.mark.parametrize("missing", ["BitArray", "DataBin"])
def test_version_guard_detects_missing_primitive_container_attributes(monkeypatch, missing):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives.containers":
                from types import SimpleNamespace

                if missing == "BitArray":
                    return SimpleNamespace(DataBin=object())
                return SimpleNamespace(BitArray=object())
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(
            ImportError,
            match=rf"qiskit\.primitives\.containers\.{missing}",
        ):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_observables_array_module(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives.containers.observables_array":
                raise ImportError("missing ObservablesArray")
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(
            ImportError, match="qiskit.primitives\\.containers.observables_array.ObservablesArray"
        ):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_observables_array_attribute(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.primitives.containers.observables_array":
                from types import SimpleNamespace

                return SimpleNamespace()
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(
            ImportError, match="qiskit.primitives\\.containers.observables_array.ObservablesArray"
        ):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)


def test_version_guard_detects_missing_ifelse(monkeypatch):
    module = importlib.reload(version_guard)
    try:
        real_import = importlib.import_module

        def _fake_import(name, package=None):
            if name == "qiskit.circuit":
                from types import SimpleNamespace

                return SimpleNamespace()
            return real_import(name, package)

        monkeypatch.setattr(module, "import_module", _fake_import)
        with pytest.raises(ImportError, match="IfElseOp"):
            module.ensure_supported_qiskit_version(installed_version="2.0.0")
    finally:
        importlib.reload(version_guard)
