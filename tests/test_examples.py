from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import (
    StatevectorEstimator as _StatevectorEstimator,
    StatevectorSampler as _StatevectorSampler,
)
from qiskit.quantum_info import Pauli

from qiskit_inspect import (
    CircuitDebugger,
    ascii_histogram,
    counts_to_dataframe,
    expectations_to_dataframe,
    probabilities_to_dataframe,
    trace_counts_with_sampler,
    trace_expectations_with_estimator,
    trace_expectations_with_statevector,
    trace_marginal_probabilities_with_sampler,
    trace_marginal_probabilities_with_statevector,
    trace_probabilities_with_sampler,
    trace_probabilities_with_statevector_exact,
    trace_records_to_dataframe,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str):
    """Load an example module by file name."""

    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # pragma: no cover - safety net for mypy
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seeded_sampler(*args, **kwargs):
    kwargs.setdefault("seed", 123)
    return _StatevectorSampler(*args, **kwargs)


def _seeded_estimator(*args, **kwargs):
    kwargs.setdefault("seed", 123)
    return _StatevectorEstimator(*args, **kwargs)


def _patch_rng(monkeypatch: pytest.MonkeyPatch, module, *, sampler: bool = True, estimator: bool = False):
    """Patch sampler/estimator constructors inside an example for determinism."""

    if sampler and hasattr(module, "StatevectorSampler"):
        monkeypatch.setattr(module, "StatevectorSampler", _seeded_sampler)
    if estimator and hasattr(module, "StatevectorEstimator"):
        monkeypatch.setattr(module, "StatevectorEstimator", _seeded_estimator)


def _sorted_dict_str(data: dict) -> str:
    return str(dict(sorted(data.items())))


def _assert_output(capsys, expected_lines: Iterable[str]):
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == list(expected_lines)


def test_bell_backend_trace(capsys, monkeypatch):
    module = _load_example("bell_backend_trace")
    _patch_rng(monkeypatch, module)

    qc = module.build_circuit()
    sampler = _seeded_sampler(default_shots=4096)
    sampled = trace_probabilities_with_sampler(qc, sampler)
    exact = trace_probabilities_with_statevector_exact(qc)

    monkeypatch.setattr(module, "trace_probabilities_with_aer", lambda *args, **kwargs: exact)

    module.main()
    captured = capsys.readouterr().out.strip().splitlines()

    expected = [
        f"[SamplerV2] Step {i}: {_sorted_dict_str(p)}" for i, p in enumerate(sampled, 1)
    ]
    expected.extend(f"[Aer] Step {i}: {_sorted_dict_str(p)}" for i, p in enumerate(exact, 1))
    assert captured == expected


def test_breakpoints_demo(capsys):
    module = _load_example("breakpoints_demo")
    module.main()
    _assert_output(capsys, ["Stop at cx: cx 2", "Stop at step: z 4"])


def test_compare_methods(capsys, monkeypatch):
    module = _load_example("compare_methods")

    qc = module.build_circuit()
    exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    marg = trace_marginal_probabilities_with_statevector(qc, [2, 0], include_initial=True)
    monkeypatch.setattr(module, "trace_probabilities_with_aer", lambda *args, **kwargs: exact[1:])

    module.main()
    out = capsys.readouterr().out

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Circuit:\n", qc)
        print("\nExact probabilities (include_initial=True):")
        for i, p in enumerate(exact):
            print(f"step {i}: {p}")
        print("\nExact marginal probs for qubits [2,0] (include_initial=True):")
        for i, p in enumerate(marg):
            print(f"step {i}: {p}")
        print("\nAer-sampled probabilities (shots=8192):")
        for i, p in enumerate(exact[1:]):
            print(f"prefix {i + 1}: {p}")

    expected = buf.getvalue()
    assert out == expected


def test_custom_condition_evaluator(capsys):
    module = _load_example("custom_condition_evaluator")
    module.main()
    _assert_output(
        capsys,
        [
            "0 None [None, None]",
            "1 h [None, None]",
            "2 measure [0, None]",
            "3 reset [0, None]",
            "4 measure [0, 0]",
            "5 if_else [0, 0]",
        ],
    )


def test_debugger_trace_walkthrough(capsys):
    module = _load_example("debugger_trace_walkthrough")
    module.main()
    out = capsys.readouterr().out

    qc = module.build_circuit()
    assert out.startswith("Circuit:\n")
    assert str(qc) in out
    for snippet in [
        "=== Structured trace (include_initial=True) ===",
        "step  0: None                  bits=xx  state=1.0000|00>",
        "step  1: h                     bits=xx  state=0.7071|00> + 0.7071|01>",
        "step  2: cx                    bits=xx  state=0.7071|00> + 0.7071|11>",
        "step  3: measure               bits=0x  state=1.0000|00>",
        "step  4: if_else               bits=0x  state=1.0000|00>",
        "step  5: ry                    bits=0x  state=0.9394|00> + 0.3429|10>",
        "step  6: measure               bits=00  state=1.0000|00>",
        "=== Flattened control flow with markers ===",
        "step  4: -                     marker=enter_if_then",
        "step  6: -                     marker=exit_if_then",
        "=== Breakpoints ===",
        "First RY instruction: ry",
        "Statevector after step 4: 1.0000|00>",
        "Top amplitudes at step 4: [('00', np.complex128(1+0j)), ('01', np.complex128(0j))]",
    ]:
        assert snippet in out


def test_expectations_and_exports(tmp_path, capsys, monkeypatch):
    module = _load_example("expectations_and_exports")
    _patch_rng(monkeypatch, module, estimator=True)
    monkeypatch.chdir(tmp_path)

    qc = module.build_expectation_circuit()
    theta = qc.parameters[0]
    theta_value = float(np.pi / 5)
    theta_binding = {theta: theta_value}
    observables = [("<Z>", Pauli("Z"), [0])]

    statevector_rows = trace_expectations_with_statevector(
        qc, observables, include_initial=True, parameter_values=[theta_value]
    )
    estimator_rows = trace_expectations_with_estimator(
        qc,
        observables,
        estimator=_seeded_estimator(),
        include_initial=True,
        parameter_values=[{theta: 0.0}, {theta: float(np.pi / 2)}],
    )

    qc_for_debug = qc.assign_parameters(theta_binding, inplace=False)
    qc_for_debug.measure_all()
    debugger = CircuitDebugger(qc_for_debug)
    trace_head = trace_records_to_dataframe(
        debugger.trace(include_initial=True),
        classical_bit_columns=["c0"],
        include_pre_measurement=True,
    ).head()

    probs_tail = probabilities_to_dataframe(
        trace_probabilities_with_statevector_exact(
            qc, include_initial=True, parameter_values=[theta_value]
        )
    ).tail()

    sampler_for_counts = _seeded_sampler()
    qc_for_counts = qc.assign_parameters(theta_binding, inplace=False)
    qc_for_counts.measure_all()
    counts_tail = counts_to_dataframe(
        trace_counts_with_sampler(qc_for_counts, sampler_for_counts)
    ).tail()

    expectations_tail = expectations_to_dataframe(statevector_rows).tail()

    module.main()
    out = capsys.readouterr().out

    output_dir = Path("expectations_output")
    assert output_dir.exists()

    statevector_csv = output_dir / "statevector.csv"
    estimator_csv = output_dir / "estimator.csv"
    trace_csv = output_dir / "trace.csv"

    assert statevector_csv.exists()
    assert estimator_csv.exists()
    assert (output_dir / "statevector.json").exists()
    assert (output_dir / "estimator.json").exists()
    assert trace_csv.exists()
    assert (output_dir / "trace.json").exists()

    statevector_df = pd.read_csv(statevector_csv)
    estimator_df = pd.read_csv(estimator_csv)
    trace_df = pd.read_csv(trace_csv)

    expected_statevector_df = expectations_to_dataframe(statevector_rows)
    expected_estimator_df = expectations_to_dataframe(estimator_rows)

    assert np.allclose(
        statevector_df[["<Z>"]].to_numpy(),
        expected_statevector_df[["<Z>"]].to_numpy(),
    )
    assert np.allclose(
        estimator_df[["<Z>"]].to_numpy(),
        expected_estimator_df[["<Z>"]].to_numpy(),
    )
    assert not trace_df.empty

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Trace DataFrame:\n", trace_head)
        print("Probabilities DataFrame:\n", probs_tail)
        print("Counts DataFrame:\n", counts_tail)
        print("Expectations DataFrame:\n", expectations_tail)

    expected_text = buf.getvalue()
    assert out == expected_text


def test_marginal_histograms(capsys, monkeypatch):
    module = _load_example("marginal_histograms")
    _patch_rng(monkeypatch, module)

    qc = module.build_circuit()
    sampler = _seeded_sampler(default_shots=2048)
    probs_each = trace_marginal_probabilities_with_sampler(
        qc,
        sampler,
        qubits=[0],
        shots=2048,
        add_measure_for_qubits=True,
    )

    calls = []

    def fake_plot_histogram(data, **kwargs):
        calls.append((data, kwargs))
        return "figure"

    monkeypatch.setattr(module, "plot_histogram", fake_plot_histogram)

    module.main()
    captured = capsys.readouterr().out

    expected = "".join(
        f"Prefix {i}\n{ascii_histogram(p, width=30)}\n\n" for i, p in enumerate(probs_each, 1)
    )

    assert captured == expected
    assert calls == [
        (
            probs_each[-1],
            {"title": "Marginal on q0 (final prefix)"},
        )
    ]


def test_parameter_broadcasting(capsys, monkeypatch):
    module = _load_example("parameter_broadcasting")
    _patch_rng(monkeypatch, module)

    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    sampler = _seeded_sampler(default_shots=1024)
    formats = {
        "dict": {theta: np.pi / 8},
        "list-of-dicts": [{theta: np.pi / 4}],
        "list": [0.0, np.pi],
        "sampler-style": [[np.pi / 3]],
    }

    expected_lines = []
    for label, params in formats.items():
        sampled = trace_probabilities_with_sampler(
            qc,
            sampler,
            shots=1024,
            parameter_values=params,
        )
        expected_lines.append(f"{label} bindings (last prefix): {sampled[-1]}")

    exact = trace_probabilities_with_statevector_exact(
        qc,
        include_initial=True,
        parameter_values=[np.pi / 8],
    )
    expected_lines.append(f"Statevector exact probabilities (last prefix): {exact[-1]}")

    module.main()
    _assert_output(capsys, expected_lines)


def test_plot_histogram_example(monkeypatch):
    module = _load_example("plot_histogram_example")
    calls = []

    def fake_plot_histogram(data, **kwargs):
        calls.append((data, kwargs))
        return "figure"

    monkeypatch.setattr(module, "plot_histogram", fake_plot_histogram)
    module.main()
    assert calls == [({"00": 0.48, "11": 0.52}, {"title": "Bell distribution"})]


def test_probabilities_and_counts_walkthrough(capsys, monkeypatch):
    module = _load_example("probabilities_and_counts_walkthrough")
    _patch_rng(monkeypatch, module)

    qc = module.build_circuit()
    sampler = _seeded_sampler(default_shots=2048)
    exact = trace_probabilities_with_statevector_exact(qc, include_initial=True)
    sampled = trace_probabilities_with_sampler(qc, sampler, shots=2048)
    counts = trace_counts_with_sampler(qc, sampler, shots=2048)
    marg_exact = trace_marginal_probabilities_with_statevector(qc, qubits=[1], include_initial=True)
    marg_sampled = trace_marginal_probabilities_with_sampler(
        qc,
        sampler,
        qubits=[1],
        shots=2048,
        add_measure_for_qubits=True,
    )

    monkeypatch.setattr(module, "trace_probabilities_with_aer", lambda *args, **kwargs: exact[1:])

    module.main()
    out = capsys.readouterr().out

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"Exact last-prefix probabilities: {_sorted_dict_str(exact[-1])}")
        print(f"Sampler last-prefix probabilities: {_sorted_dict_str(sampled[-1])}")
        print(f"Sampler last-prefix counts: {_sorted_dict_str(counts[-1])}")
        print("ASCII histogram:\n", ascii_histogram(exact[-1]))
        print(f"Exact marginals for qubit 1: {marg_exact[-1]}")
        print(f"Sampled marginals for qubit 1: {marg_sampled[-1]}")
        print(f"Aer last-prefix probabilities: {_sorted_dict_str(exact[-1])}")

    expected = buf.getvalue()
    assert out == expected


def test_teleportation_ifelse(capsys):
    module = _load_example("teleportation_ifelse")
    module.main()
    out = capsys.readouterr().out

    buf = io.StringIO()
    with redirect_stdout(buf):
        dbg = module.CircuitDebugger(module.build_circuit(), seed=123)
        for rec in dbg.trace():
            print(
                f"{rec.step_index:02d} {rec.instruction or 'init':>10}: "
                f"{module.pretty_ket(rec.state)}   C={rec.classical_bits}"
            )

    expected = buf.getvalue()
    assert out == expected
