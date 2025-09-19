# qiskit-inspect Docs

Welcome! qiskit-inspect helps you understand and validate Qiskit 2.0+ circuits with an exact statevector debugger, per-prefix probability tracing (sampler/Aer/exact), and testing-friendly assertions. The package requires a Qiskit 2.0 or newer installation.

## Start here

- Getting started: [Installation and quick tour](./getting-started.md)
- Examples: [Copy‑pasteable scripts](./examples.md) covering debugger, tracing, exports, and testing

## Core guides

- How it works: [Architecture and algorithms](./how-it-works.md)
- Bitstrings & probabilities: [Ordering, marginals, sparsity](./bitstrings-and-probabilities.md)

## API and references

- API reference: [Classes and functions](./api-reference.md)

## Practical resources

- Cookbook: [Common recipes](./cookbook.md)
- Troubleshooting: [Fix common issues](./troubleshooting.md)
- Performance & limits: [What scales, what doesn’t](./performance-and-limits.md)

Optional dependencies (Aer, IBM Runtime, pandas, Matplotlib/Seaborn) are
available via ``pip install -e .[extra]`` so you can enable only the workflows
you need while experimenting locally.
