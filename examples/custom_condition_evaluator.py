"""Override classical expression handling when debugging control flow."""

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit_inspect import CircuitDebugger


def build_circuit() -> tuple[QuantumCircuit, ClassicalRegister, ClassicalRegister]:
    qc = QuantumCircuit(1)
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc.add_register(a, b)

    qc.h(0)
    qc.measure(0, a[0])
    qc.reset(0)
    qc.measure(0, b[0])

    # Insert a condition using one of the measured bits.  At runtime Qiskit
    # expects ``if_test`` to receive a ``(bit, value)`` tuple, but the debugger's
    # custom evaluator will ignore the declared condition and choose a branch
    # based on the XOR of registers ``a`` and ``b``.
    with qc.if_test((a[0], 1)):
        pass

    return qc, a, b


def make_xor_evaluator(reg_a: ClassicalRegister, reg_b: ClassicalRegister):
    def xor_evaluator(cond_obj, classical_bits, circuit):
        # Ignore cond_obj and compute XOR of the first two bits
        idx_a = circuit.find_bit(reg_a[0]).index
        idx_b = circuit.find_bit(reg_b[0]).index
        a_val = int(classical_bits[idx_a] or 0)
        b_val = int(classical_bits[idx_b] or 0)
        return bool(a_val ^ b_val)

    return xor_evaluator


def main() -> None:
    qc, reg_a, reg_b = build_circuit()
    evaluator = make_xor_evaluator(reg_a, reg_b)
    dbg = CircuitDebugger(qc, condition_evaluator=evaluator)
    for rec in dbg.trace():
        print(rec.step_index, rec.instruction, rec.classical_bits)


if __name__ == "__main__":
    main()
