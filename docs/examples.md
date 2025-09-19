# Examples

Runnable scripts live in the repository's [`examples/`](../examples/) directory.
All snippets assume a Qiskit 2.0+ environment. Launch them with `python` to see
representative output.

## Debugger trace walkthrough

Print structured traces, flattened control flow, and breakpoint snapshots. The
script also enables trace logging so you can see classical bit ordering in
real-time:

```bash
python examples/debugger_trace_walkthrough.py
```

## Probabilities and counts walkthrough

Compare exact prefix probabilities with sampler counts, marginals, and optional
Aer sampling:

```bash
python examples/probabilities_and_counts_walkthrough.py
```

Aer helpers are guarded—if `qiskit-aer` is not installed, the script reports the
missing dependency and continues.

## Expectations, DataFrames, and exports

Generate per-prefix expectation values, materialize pandas DataFrames, and write
CSV/JSON artifacts to disk:

```bash
python examples/expectations_and_exports.py
```

The script creates an `expectations_output/` directory populated with trace and
expectation exports.

## Parameter broadcasting showcase

See every supported parameter binding style (dicts, lists, list-of-dicts, and
sampler-style double lists) in action:

```bash
python examples/parameter_broadcasting.py
```

## Testing helpers

Run the assertion helpers against sampled and exact distributions:

```bash
python examples/testing_helpers.py
```

## Additional focused demos

- `bell_backend_trace.py` – Minimal prefix tracing on a Bell circuit.
- `breakpoints_demo.py` – Quick look at `run_until_*` helpers.
- `compare_methods.py` – Exact probabilities, marginals, and Aer sampling.
- `custom_condition_evaluator.py` – Override classical condition evaluation.
- `marginal_histograms.py` – Plot marginal distributions with Qiskit's Matplotlib helpers.
- `plot_histogram_example.py` – Uses `qiskit.visualization.plot_histogram` directly (requires Matplotlib).
- `teleportation_ifelse.py` – Debug an If/Else-controlled teleportation circuit.

## Optional dependencies by example

- `probabilities_and_counts_walkthrough.py`, `bell_backend_trace.py`, `compare_methods.py`
  - Optional: `qiskit-aer` (for Aer-based sampling).
- `expectations_and_exports.py`
  - Optional: `pandas` (DataFrame printing) and `qiskit-aer` (not required but compatible).
- `marginal_histograms.py`, `plot_histogram_example.py`
  - Optional: `matplotlib` (visualization via Qiskit's plotting tools).

Every script degrades gracefully when optional extras are missing—warnings are
printed instead of raising exceptions.
