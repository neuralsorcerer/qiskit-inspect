"""Demonstrate debugger breakpoints that stop on operations or indices."""

from qiskit import QuantumCircuit

from qiskit_inspect import CircuitDebugger


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.z(2)
    qc.h(2)
    return qc


def main() -> None:
    dbg = CircuitDebugger(build_circuit())

    # Stop at first cx
    snap1 = dbg.run_until_op("cx")
    print("Stop at cx:", snap1[-1].instruction, snap1[-1].step_index)

    # Continue to step 4
    snap2 = dbg.run_until_index(4)
    print("Stop at step:", snap2[-1].instruction, snap2[-1].step_index)


if __name__ == "__main__":
    main()
