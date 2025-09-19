# Performance & Limits

This page summarizes performance characteristics and practical limits.

## Complexity

- Exact statevector stepping: O(2^n) memory and time.
- Sampling (SamplerV2 / Aer): scales with shots and circuit depth; CPU-bound.
- Marginal post-processing: proportional to number of measured qubits and outcomes retained.
- Control-flow flattening: linear in the number of executed instructions, even
  when tracing nested ``ForLoopOp``/``WhileLoopOp``/``SwitchCaseOp`` blocks.

## Tips

- Prefer exact methods for small circuits (<~20 qubits) and correctness checks.
- For larger circuits, use sampler-based methods, and target marginals of interest instead of full distributions.
- Reuse samplers and avoid recreating primitives inside loops.
- Enable ``flatten_control_flow=True`` only when you need per-iteration
  snapshots; otherwise the debugger skips replaying loop bodies for every
  branch marker to minimize memory pressure.

## Numerical considerations

- Probabilities are rounded to 15 decimals to avoid tiny floating artifacts.
- Aer comparisons should use a tolerance that scales with 1/sqrt(shots).
- ``trace_counts_with_sampler`` normalizes heterogeneous key types so repeated
  runs produce stable dictionary layouts suitable for CSV exports and
  comparisons.

## Known limits

- No transpilation: circuits are traced as-authored; backend noise models are not simulated unless you configure Aer accordingly.
- Control-flow: ``IfElseOp``, ``ForLoopOp``, ``WhileLoopOp``, and ``SwitchCaseOp``
  are supported. Infinite ``WhileLoopOp`` bodies raise an error once the
  configured iteration guard is exceeded.
