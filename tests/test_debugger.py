import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import CircuitInstruction, Instruction, Parameter
from qiskit.circuit.controlflow import (
    CASE_DEFAULT,
    BreakLoopOp,
    ContinueLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.quantum_info import Statevector

from qiskit_inspect import CircuitDebugger, TraceRecord, assert_state_equiv


def test_bell_equivalence():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    dbg = CircuitDebugger(qc)
    final = dbg.run_all().state
    assert_state_equiv(final, Statevector.from_instruction(qc))


def test_initial_state_rejects_mismatched_statevector():
    qc = QuantumCircuit(1)
    bad_state = Statevector.from_label("00")
    with pytest.raises(ValueError, match="acts on 2 qubits"):
        CircuitDebugger(qc, initial_state=bad_state)


def test_initial_state_rejects_non_power_of_two_vector():
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="not a power of two"):
        CircuitDebugger(qc, initial_state=[1, 0, 0])


def test_initial_state_bitstring_requires_exact_qubit_width():
    qc = QuantumCircuit(3)
    with pytest.raises(ValueError, match="exactly 3 qubits"):
        CircuitDebugger(qc, initial_state="01")


def test_initial_state_bitstring_sanitization_and_initialization():
    qc = QuantumCircuit(3)
    dbg = CircuitDebugger(qc, initial_state=" 1_0 1 ")
    final_state = dbg.run_all().state

    assert_state_equiv(final_state, Statevector.from_label("101"))


def test_initial_state_bitstring_allows_empty_for_zero_qubits():
    qc = QuantumCircuit(0)
    dbg = CircuitDebugger(qc, initial_state="")
    final_state = dbg.run_all().state

    assert_state_equiv(final_state, Statevector([1.0]))


def test_ifelse_and_measure():
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(1)
    qc.add_register(cr)
    qc.h(0)
    qc.measure(0, cr[0])
    with qc.if_test((cr, 1)):  # if measured 1, apply X
        qc.x(0)
    dbg = CircuitDebugger(qc, seed=7)
    recs = dbg.trace()
    assert recs[-1].classical_bits[0] in (0, 1)


def test_for_loop_executes_all_iterations():
    qc = QuantumCircuit(1, 1)

    body = QuantumCircuit(1, 1)
    body.x(0)
    body.measure(0, 0)

    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    rec = dbg.trace(include_initial=False)[-1]

    assert rec.instruction == "for_loop"
    assert rec.classical_bits == [0]
    assert_state_equiv(rec.state, Statevector.from_label("0"))


def test_for_loop_parameter_binding():
    theta_vals = [0.1, 0.2]
    param = Parameter("theta")
    body = QuantumCircuit(1)
    body.ry(param, 0)

    qc = QuantumCircuit(1)
    loop = ForLoopOp(theta_vals, param, body)
    qc.append(loop, qc.qubits, [])

    dbg = CircuitDebugger(qc)
    final = dbg.run_all().state

    expected = Statevector.from_label("0")
    for value in theta_vals:
        expected = expected.evolve(body.assign_parameters({param: value}))
    assert_state_equiv(final, expected)


def test_for_loop_break_exits_early():
    qc = QuantumCircuit(1)

    body = QuantumCircuit(1)
    body.x(0)
    body.append(BreakLoopOp(0, 0), [], [])
    body.x(0)  # should not execute after break

    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, [])

    dbg = CircuitDebugger(qc)
    final = dbg.run_all().state

    assert_state_equiv(final, Statevector.from_label("1"))


def test_for_loop_continue_skips_rest_of_iteration():
    qc = QuantumCircuit(1, 1)

    body = QuantumCircuit(1, 1)
    body.x(0)
    body.append(ContinueLoopOp(0, 0), [], [])
    body.measure(0, 0)  # skipped because of continue

    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    final_record = dbg.run_all()

    assert final_record.classical_bits == [None]
    assert_state_equiv(final_record.state, Statevector.from_label("0"))


def test_trace_markers_include_loop_iterations():
    qc = QuantumCircuit(1)

    body = QuantumCircuit(1)
    body.x(0)

    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, [])

    dbg = CircuitDebugger(qc)
    names = [rec.instruction for rec in dbg.trace(include_markers=True)]

    assert names.count("for_loop") == 1
    loop_markers = [name for name in names if name and name.startswith("for_iter[")]
    assert loop_markers == ["for_iter[0:0]", "for_iter[1:1]"]


def test_while_loop_runs_until_condition_false():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.x(0)
    body.measure(0, 0)

    loop = WhileLoopOp((qc.clbits[0], 1), body)
    qc.append(loop, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    rec = dbg.trace(include_initial=False)[-1]

    assert rec.instruction == "while_loop"
    assert rec.classical_bits == [0]
    assert_state_equiv(rec.state, Statevector.from_label("0"))


def test_while_loop_detects_non_terminating():
    qc = QuantumCircuit(1, 1)

    body = QuantumCircuit(1, 1)
    body.x(0)

    loop = WhileLoopOp((qc.clbits[0], 0), body)
    qc.append(loop, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)

    with pytest.raises(RuntimeError, match="WhileLoopOp exceeded maximum iterations"):
        dbg.run_all()


def test_while_loop_condition_evaluator_invoked():
    qc = QuantumCircuit(1, 1)
    body = QuantumCircuit(1, 1)
    body.append(BreakLoopOp(0, 0), [], [])

    loop = WhileLoopOp((qc.clbits[0], 0), body)
    sentinel = object()
    loop.condition = sentinel
    qc.append(loop, qc.qubits, qc.clbits)

    seen = []

    def evaluator(obj, classical_bits, circuit):
        seen.append(obj)
        return False

    dbg = CircuitDebugger(qc, condition_evaluator=evaluator)
    dbg.run_all()

    assert seen == [sentinel]


def test_trace_flatten_emits_nested_operations_and_markers():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)

    body = QuantumCircuit(1)
    body.x(0)
    loop = ForLoopOp(range(2), None, body)
    qc.append(loop, qc.qubits, [])

    dbg = CircuitDebugger(qc)
    recs = dbg.trace(include_initial=True, include_markers=True, flatten_control_flow=True)

    names = [rec.instruction for rec in recs if rec.instruction]
    assert names == [
        "h",
        "measure",
        "if_else",
        "for_iter[0:0]",
        "x",
        "for_iter[1:1]",
        "x",
        "for_loop",
    ]


def test_trace_flatten_without_markers_omits_loop_annotations():
    qc = QuantumCircuit(1)

    body = QuantumCircuit(1)
    body.x(0)
    loop = ForLoopOp(range(3), None, body)
    qc.append(loop, qc.qubits, [])

    dbg = CircuitDebugger(qc)
    recs = dbg.trace(include_initial=False, flatten_control_flow=True)

    names = [rec.instruction for rec in recs]
    assert names == ["x", "x", "x", "for_loop"]


def test_trace_flatten_records_while_iterations():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)

    body = QuantumCircuit(1, 1)
    body.x(0)
    body.measure(0, 0)

    loop = WhileLoopOp((qc.clbits[0], 1), body)
    qc.append(loop, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    recs = dbg.trace(include_initial=False, include_markers=True, flatten_control_flow=True)

    names = [rec.instruction for rec in recs]
    assert names == ["x", "measure", "while_iter[0]", "x", "measure", "while_loop"]


def test_switch_case_executes_matching_branch():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    case_zero = QuantumCircuit(1, 1)
    case_zero.x(0)
    case_one = QuantumCircuit(1, 1)
    case_one.z(0)
    case_default = QuantumCircuit(1, 1)
    case_default.sx(0)

    switch = SwitchCaseOp(
        qc.cregs[0], [(0, case_zero), (1, case_one), (CASE_DEFAULT, case_default)]
    )
    qc.append(switch, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    rec = dbg.trace(include_initial=False)[-1]

    assert rec.instruction == "switch_case"
    assert_state_equiv(rec.state, Statevector.from_label("1"))


def test_c_if_instruction_respected():
    qc = QuantumCircuit(1, 1)
    meas_inst = Instruction("measure", 1, 1, [])
    qc.append(meas_inst, [qc.qubits[0]], [qc.clbits[0]])
    x_inst = Instruction("x", 1, 0, [])
    x_inst.condition = (qc.cregs[0], 1)
    qc.append(x_inst, [qc.qubits[0]], [])

    dbg = CircuitDebugger(qc)
    recs = dbg.trace()
    final_state = recs[-1].state

    assert_state_equiv(final_state, Statevector.from_label("0"))
    assert recs[-1].classical_bits[0] == 0


def test_condition_evaluator_handles_instruction_condition_objects():
    class Sentinel:
        pass

    cond_obj = Sentinel()
    qc = QuantumCircuit(1, 1)
    qc.x(0)

    seen = []

    def evaluator(obj, classical_bits, circuit):
        seen.append(obj)
        assert obj is cond_obj
        return True

    dbg = CircuitDebugger(qc, condition_evaluator=evaluator)
    ci = dbg.instructions[0]
    op = ci.operation.to_mutable()
    op.condition = cond_obj
    mutated = CircuitInstruction(op, ci.qubits, ci.clbits)
    dbg.instructions[0] = mutated
    dbg.circuit.data[0] = mutated
    final_state = dbg.run_all().state

    assert seen  # evaluator invoked
    assert_state_equiv(final_state, Statevector.from_label("1"))


def test_run_until_variants():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(1)
    dbg = CircuitDebugger(qc)
    # until op name
    recs = dbg.run_until_op("cx")
    assert recs[-1].instruction == "cx"
    # until index
    recs2 = dbg.run_until_index(2)
    assert recs2[-1].step_index == 2


def test_multi_qubit_measure_updates_all_classical_bits():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    measure_inst = Instruction("measure", 2, 2, [])
    qc.append(measure_inst, qc.qubits, qc.clbits)

    dbg = CircuitDebugger(qc)
    recs = dbg.trace()
    final_bits = recs[-1].classical_bits
    assert final_bits == [1, 0]


def test_multi_qubit_measure_respects_qubit_classical_pairing_order():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(2)
    measure_inst = Instruction("measure", 3, 3, [])
    qc.append(
        measure_inst,
        (qc.qubits[2], qc.qubits[0], qc.qubits[1]),
        (qc.clbits[1], qc.clbits[0], qc.clbits[2]),
    )

    dbg = CircuitDebugger(qc)
    final_bits = dbg.trace()[-1].classical_bits
    assert final_bits == [1, 1, 0]


def test_initialize_instruction_supported():
    qc = QuantumCircuit(1)
    qc.initialize([0, 1], 0)

    dbg = CircuitDebugger(qc)
    final_state = dbg.run_all().state

    assert_state_equiv(final_state, Statevector.from_label("1"))


def test_initialize_multiple_qubits_supported():
    qc = QuantumCircuit(2)
    qc.initialize([0, 1, 0, 0], [0, 1])

    dbg = CircuitDebugger(qc)
    final_state = dbg.run_all().state

    assert_state_equiv(final_state, Statevector.from_instruction(qc))


def test_initialize_inside_control_flow_branch():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.initialize([1, 0], 0)

    dbg = CircuitDebugger(qc, seed=1)
    final_record = dbg.run_all()

    assert final_record.classical_bits == [1]
    assert_state_equiv(final_record.state, Statevector.from_label("0"))


def test_trace_record_to_dict_normalizes_probability_keys(monkeypatch):
    state = Statevector.from_label("0")
    record = TraceRecord(0, None, state, [None])

    def fake_probs():
        return {(1,): 0.6, 0: 0.4}

    monkeypatch.setattr(record.state, "probabilities_dict", fake_probs)
    serialized = record.to_dict(state_format="probs")
    assert serialized["state"] == {"1": 0.6, "0": 0.4}


def test_trace_record_to_dict_zero_qubits():
    record = TraceRecord(0, None, Statevector([1.0]), [])

    serialized = record.to_dict(state_format="probs")

    assert serialized["state"] == {"": 1.0}
