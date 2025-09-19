from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.quantum_info import Statevector

from qiskit_inspect import CircuitDebugger, assert_state_equiv


class Bin:
    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs


class Un:
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand


class Bits:
    def __init__(self, bits):
        self.bits = bits


class Val:
    def __init__(self, value):
        self.value = value


def test_builtin_evaluator_binary_ops():
    qc = QuantumCircuit(1)
    c = ClassicalRegister(2, "c")
    qc.add_register(c)
    qc.h(0)
    qc.measure(0, c[0])
    qc.reset(0)
    qc.measure(0, c[1])

    # Build a placeholder IfElseOp using a tuple, but the debugger will evaluate our custom object
    with qc.if_test((c, 0)):
        qc.x(0)

    dbg = CircuitDebugger(qc)
    # Override internal evaluation target directly (simulate condition tree)
    conds = [
        (Bin(Val(1), "==", Val(1)), True),
        (Bin(Val(1), "!=", Val(0)), True),
        (Bin(Val(0), "and", Val(1)), False),
        (Bin(Val(1), "or", Val(0)), True),
        (Bin(Val(1), "^", Val(1)), False),
        (Bin(Val(0), "^", Val(1)), True),
        (Bin(Val(1), ">", Val(0)), True),
        (Bin(1, "<=", 1), True),
        (Bin(Bin(Val(5), "mod", Val(2)), "==", Val(1)), True),
        (Bin(Bin(Val(5), "%", Val(2)), "==", Val(1)), True),
        (Bin(Bin(Val(2), "pow", Val(3)), "==", Val(8)), True),
        (Bin(Bin(Val(2), "**", Val(3)), "==", Val(8)), True),
        (Bin(Bin(Val(5.0), "div", Val(2.0)), "==", Val(2.5)), True),
        (Bin(Bin(Val(5), "/", Val(2)), "==", Val(2.5)), True),
        (Bin(Bin(Val(5), "truediv", Val(2)), "==", Val(2.5)), True),
        (Bin(Bin(Val(3), "shift_left", Val(1)), "==", Val(6)), True),
        (Bin(Bin(Val(3), "<<", Val(1)), "==", Val(6)), True),
        (Bin(Bin(Val(4), "shift_right", Val(1)), "==", Val(2)), True),
        (Bin(Bin(Val(4), ">>", Val(1)), "==", Val(2)), True),
        (Bin(Bin(Val(7), "floordiv", Val(2)), "==", Val(3)), True),
        (Bin(Bin(Val(7), "//", Val(2)), "==", Val(3)), True),
    ]

    for obj, expected in conds:
        assert dbg._eval_condition_object(obj) is expected


def test_builtin_evaluator_unary_and_bits():
    qc = QuantumCircuit(1)
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc.add_register(a, b)
    qc.h(0)
    qc.measure(0, a[0])
    qc.reset(0)
    qc.measure(0, b[0])

    with qc.if_test((a, 0)):
        pass

    dbg = CircuitDebugger(qc)
    # Force classical bits to specific values
    dbg.classical_bits = [1, 0]  # a=1, b=0

    # Unary not
    assert dbg._eval_condition_object(Un("not", Val(0))) is True
    assert dbg._eval_condition_object(Un("!", Val(1))) is False

    # Bits container
    assert dbg._eval_condition_object(Bits([a[0], b[0]])) is True  # any bit set


def test_evaluator_unsupported_operator_returns_false():
    qc = QuantumCircuit(1)
    a = ClassicalRegister(1, "a")
    qc.add_register(a)
    with qc.if_test((a, 0)):
        pass
    dbg = CircuitDebugger(qc)
    # Unknown binary operator
    assert dbg._eval_condition_object(Bin(1, "weirdop", 2)) is False
    # Unknown unary operator: treat as identity fallback -> returns True since operand is 1
    # Here we expect not recognized unary to default to identity (which our code does), so True
    assert dbg._eval_condition_object(Un("weirdun", 1)) is True


def test_builtin_evaluator_handles_qiskit_expression_nodes():
    qc = QuantumCircuit(1, 3)
    qc.x(0)
    qc.measure(0, qc.clbits[0])
    qc.reset(0)
    qc.measure(0, qc.clbits[1])

    cond = expr.bit_xor(expr.Var(qc.clbits[0], types.Bool()), expr.Var(qc.clbits[1], types.Bool()))
    with qc.if_test(cond):
        qc.x(0)
    qc.measure(0, qc.clbits[2])

    dbg = CircuitDebugger(qc)
    final = dbg.run_all()

    assert final.classical_bits[:2] == [1, 0]
    assert final.classical_bits[2] == 1
    assert_state_equiv(final.state, Statevector.from_label("1"))


def test_builtin_evaluator_handles_division_and_indexing():
    qc = QuantumCircuit(1, 3)
    dbg = CircuitDebugger(qc)
    dbg.classical_bits = [1, 0, 1]

    # Floating-point division should resolve using the accelerated evaluator.
    div_expr = expr.equal(expr.div(5.0, 2.0), 2.5)
    assert dbg._eval_condition_object(div_expr) is True

    # Register indexing should extract little-endian bit positions.
    register_expr = expr.Var(qc.cregs[0], types.Uint(len(qc.cregs[0])))
    high_bit = expr.index(register_expr, 2)
    low_bit = expr.index(register_expr, 1)

    assert dbg._eval_condition_object(high_bit) is True
    assert dbg._eval_condition_object(low_bit) is False


def test_condition_evaluator_overrides_ifelse_expression():
    qc = QuantumCircuit(1, 3)
    qc.x(0)
    qc.measure(0, qc.clbits[0])
    qc.reset(0)
    qc.measure(0, qc.clbits[1])

    cond = expr.bit_xor(expr.Var(qc.clbits[0], types.Bool()), expr.Var(qc.clbits[1], types.Bool()))
    with qc.if_test(cond):
        qc.x(0)
    qc.measure(0, qc.clbits[2])

    seen = []

    def evaluator(obj, classical_bits, circuit):
        seen.append(obj)
        assert obj is cond
        return False

    dbg = CircuitDebugger(qc, condition_evaluator=evaluator)
    final = dbg.run_all()

    assert seen, "custom evaluator should be invoked for expression conditions"
    assert final.classical_bits[:2] == [1, 0]
    assert final.classical_bits[2] == 0
    assert_state_equiv(final.state, Statevector.from_label("0"))
