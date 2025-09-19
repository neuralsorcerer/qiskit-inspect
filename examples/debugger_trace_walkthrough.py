from __future__ import annotations

"""Step through circuits with CircuitDebugger and trace logging.

This walkthrough demonstrates the debugger's structured trace output, optional
flattening of control flow, and breakpoint helpers that stop on specific
operations or step indices. Run the script directly to print the annotated
trace to stdout. Logging messages are sent to stderr when trace logging is
enabled.
"""

import logging

from qiskit import QuantumCircuit

from qiskit_inspect import (
    CircuitDebugger,
    enable_trace_logging,
    format_classical_bits,
    pretty_ket,
    top_amplitudes,
)


def build_circuit() -> QuantumCircuit:
    """Create a small circuit with measurements and classical control."""

    qc = QuantumCircuit(2, 2, name="debugger_demo")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    control_bit = qc.clbits[0]
    with qc.if_test((control_bit, 0)):
        qc.z(1)
    qc.ry(0.7, 1)
    qc.measure(1, 1)
    return qc


def main() -> None:
    qc = build_circuit()
    print("Circuit:\n", qc)

    # Emit INFO/DEBUG records describing classical bit order and prefix steps.
    logger = enable_trace_logging(level=logging.INFO)
    logger.info("Trace logging enabled for debugger walkthrough")

    debugger = CircuitDebugger(qc)

    print("\n=== Structured trace (include_initial=True) ===")
    for record in debugger.trace(include_initial=True):
        state = pretty_ket(record.state)
        bits = format_classical_bits(record.classical_bits)
        print(
            f"step {record.step_index:>2}: {record.instruction!s:<20}"
            f"  bits={bits}  state={state}"
        )

    print("\n=== Flattened control flow with markers ===")
    flattened = debugger.trace(
        include_initial=True,
        include_markers=True,
        flatten_control_flow=True,
    )
    marker_names = {
        "enter_if_then",
        "exit_if_then",
        "enter_if_else",
        "exit_if_else",
        "break",
        "continue",
    }
    for record in flattened:
        instr = record.instruction or ""
        is_marker = instr in marker_names or instr.startswith(
            ("for_iter[", "while_iter[", "switch_case[")
        )
        marker = instr if is_marker and instr else "-"
        op_name = "-" if is_marker else (record.instruction or "None")
        print(f"step {record.step_index:>2}: {op_name!s:<20}" f"  marker={marker}")

    print("\n=== Breakpoints ===")
    debugger.reset()
    snap_op = debugger.run_until_op("ry")
    snap_idx = debugger.run_until_index(4)
    print("First RY instruction:", snap_op[-1].instruction)
    print("Statevector after step 4:", pretty_ket(snap_idx[-1].state))
    print("Top amplitudes at step 4:", top_amplitudes(snap_idx[-1].state, k=2))


if __name__ == "__main__":
    main()
