# Troubleshooting

## Common issues and fixes

- ImportError: cannot import name ...

  - Ensure you’re on Qiskit 2.0 or newer. Reinstall dependencies, and restart your Python session.

- Aer not installed

  - Aer features are optional. Install `qiskit-aer>=0.14` or skip Aer examples/tests.

- IBM Runtime sampler unavailable

  - Optional. Install `qiskit-ibm-runtime>=0.40` and configure credentials.

- Matplotlib not installed

  - Visualization is optional. Install `matplotlib` or use ASCII histograms.

- Bitstrings appear reversed

  - Review [Bitstrings & Probabilities](./bitstrings-and-probabilities.md) for ordering rules. Enable logging via `enable_trace_logging()` and pass `debug_bit_order=True` to trace functions to print classical bit order and measured maps per prefix.

- Marginal dicts don’t include all keys

  - Outputs are sparse by design; zero-probability outcomes are omitted.

- All-zero probabilities after filtering
  - The implementation retains an argmax key to avoid empty results; consider increasing precision if needed.

## Getting help

- Check the `tests/` directory for usage patterns.
- Open an issue with your circuit snippet and expected behavior.
