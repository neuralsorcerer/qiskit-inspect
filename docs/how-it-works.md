# How it works

This page explains the architecture and core algorithms behind qiskit-inspect.
All behaviour described targets Qiskit 2.0+ semantics (SamplerV2, modern
control flow, and ``Statevector`` ``qargs`` support).

## Components

- CircuitDebugger: Steps through a circuit exactly using ``Statevector``,
  respecting measurement collapse, reset, and the full suite of classical
  control-flow operations (``IfElseOp``, ``ForLoopOp``, ``WhileLoopOp``, and
  ``SwitchCaseOp``).
- Backend tracing: Derives per-prefix probability distributions using
  SamplerV2/Aer or exact stepping.
- Assertions: Compares states and distributions with tolerances for robust
  tests.

## Exact stepping

At each instruction prefix, the debugger computes a new statevector:

- Unitary ops: apply to the current state.
- Measure: collapse using Statevector.measure semantics and record classical bits.
- Reset: prepare |0⟩ on the target qubit.
- ``IfElseOp``: choose branch based on recorded classical values (or supplied
  evaluator).
- ``ForLoopOp`` / ``WhileLoopOp``: iterate bodies by replaying the debugger for
  each loop iteration until the index set or condition terminates. Infinite
  loops are guarded with a maximum-iteration safety check.
- ``SwitchCaseOp``: evaluate the matching branch once per prefix and continue
  tracing within the chosen subcircuit.

This enables exact per-prefix probabilities by squaring amplitudes (and
rounding to 15 decimals to avoid tiny float noise).

## Sampler/Aer tracing

For sampling-based methods, the tool constructs circuits per prefix with appropriate measurements and collects counts:

- SamplerV2: extracts counts via `join_data().get_counts()` when available,
  with robust fallbacks (`data.meas.get_counts()` / `get_counts()`).
- Aer: uses qiskit-aer SamplerV2; wrappers expose both `trace_probabilities_with_aer`
  and `trace_counts_with_aer` and emit helpful errors when the optional
  dependency is missing.

Counts are either normalized to probabilities or returned raw (via
`trace_counts_with_sampler`); marginals may auto-inject per-qubit measures
where needed.

Enable `qiskit_inspect` trace logs with `enable_trace_logging()` and pass `debug_bit_order=True` to trace functions to see classical bit order and qubit→clbit maps per prefix.

## Bitstrings and marginals

- Full distributions: MSB on left corresponds to highest qubit index.
- Marginals: leftmost bit corresponds to the first qubit in the provided list; outputs are sparse (zero-probability omitted).

## Limits

- Exact methods scale as O(2^n) in memory/time.
- Sampling methods scale with shots and circuit depth; classical post-processing is modest.

See also: API Reference and Performance & Limits.
