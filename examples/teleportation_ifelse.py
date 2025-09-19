"""Debug an If/Else-controlled teleportation circuit."""

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit_inspect import CircuitDebugger, pretty_ket


def build_circuit() -> QuantumCircuit:
    # Teleport |+> from q0 to q2 with IfElseOp-based correction (Qiskit 2.0+ control flow)
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(3)
    qc.add_register(a, b)

    # Prepare state on q0
    qc.h(0)
    # Bell pair q1-q2
    qc.h(1)
    qc.cx(1, 2)
    # Bell measurement on q0-q1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, a[0])
    qc.measure(1, b[0])

    # IfElseOp via context managers
    with qc.if_test((a[0], 1)):
        qc.z(2)
    with qc.if_test((b[0], 1)):
        qc.x(2)

    return qc


def main() -> None:
    dbg = CircuitDebugger(build_circuit(), seed=123)
    for rec in dbg.trace():
        print(
            f"{rec.step_index:02d} {rec.instruction or 'init':>10}: "
            f"{pretty_ket(rec.state)}   C={rec.classical_bits}"
        )


if __name__ == "__main__":
    main()
