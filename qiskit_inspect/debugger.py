"""Deterministic statevector debugger for Qiskit circuits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

import numpy as np
from qiskit import QuantumCircuit  # type: ignore[import-untyped]
from qiskit.circuit import ClassicalRegister, Clbit, IfElseOp  # type: ignore[import-untyped]
from qiskit.circuit.controlflow import (  # type: ignore[import-untyped]
    CASE_DEFAULT,
    BreakLoopOp,
    ContinueLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.exceptions import QiskitError  # type: ignore[import-untyped]
from qiskit.quantum_info import Operator, Statevector  # type: ignore[import-untyped]

from .probabilities import normalize_probability_dict

try:  # pragma: no cover - defensive import: accelerator modules are optional
    from qiskit.circuit.classical import (  # type: ignore[import-untyped]
        expr as _classical_expr,
        types as _classical_types,
    )
except Exception:  # pragma: no cover - when classical expr support is unavailable
    _classical_expr = None  # type: ignore[assignment]
    _classical_types = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qiskit.circuit.classical import expr as _expr_typing


if _classical_expr is not None:
    _CLASSICAL_EXPR_BASE: tuple[type[Any], ...] = (_classical_expr.Expr,)
else:  # pragma: no cover - accelerated module missing
    _CLASSICAL_EXPR_BASE = ()


if _classical_expr is not None:

    class _ClassicalExprEvaluator(_classical_expr.ExprVisitor[Any]):
        """Evaluate Qiskit classical expressions using debugger state."""

        __slots__ = ("_debugger",)

        def __init__(self, debugger: "CircuitDebugger") -> None:
            self._debugger = debugger

        def evaluate(self, node: "_expr_typing.Expr") -> Any:
            return node.accept(self)

        # Visitors
        def visit_value(self, node: "_expr_typing.Value", /) -> Any:
            return node.value

        def visit_var(self, node: "_expr_typing.Var", /) -> Any:
            target = node.var
            if isinstance(target, Clbit):
                return self._debugger._bit_value(target)
            if isinstance(target, ClassicalRegister):
                return self._debugger._register_value(target)
            return self._debugger._resolve_value(target)

        def visit_unary(self, node: "_expr_typing.Unary", /) -> Any:
            operand = node.operand.accept(self)
            op = node.op

            if op == _classical_expr.Unary.Op.LOGIC_NOT:
                result = not bool(operand)
            elif op == _classical_expr.Unary.Op.BIT_NOT:
                result = ~int(operand)
            else:  # pragma: no cover - future-proof fallback
                result = operand
            return self._cast_value(result, node.type)

        def visit_binary(self, node: "_expr_typing.Binary", /) -> Any:
            left = node.left.accept(self)
            right = node.right.accept(self)
            op = node.op

            if op == _classical_expr.Binary.Op.BIT_AND:
                result = int(left) & int(right)
            elif op == _classical_expr.Binary.Op.BIT_OR:
                result = int(left) | int(right)
            elif op == _classical_expr.Binary.Op.BIT_XOR:
                result = int(left) ^ int(right)
            elif op == _classical_expr.Binary.Op.LOGIC_AND:
                result = bool(left) and bool(right)
            elif op == _classical_expr.Binary.Op.LOGIC_OR:
                result = bool(left) or bool(right)
            elif op == getattr(_classical_expr.Binary.Op, "LOGIC_XOR", None):
                result = bool(left) ^ bool(right)
            elif op == _classical_expr.Binary.Op.EQUAL:
                result = left == right
            elif op == _classical_expr.Binary.Op.NOT_EQUAL:
                result = left != right
            elif op == _classical_expr.Binary.Op.LESS:
                result = left < right
            elif op == _classical_expr.Binary.Op.LESS_EQUAL:
                result = left <= right
            elif op == _classical_expr.Binary.Op.GREATER:
                result = left > right
            elif op == _classical_expr.Binary.Op.GREATER_EQUAL:
                result = left >= right
            elif op == _classical_expr.Binary.Op.SHIFT_LEFT:
                result = int(left) << int(right)
            elif op == _classical_expr.Binary.Op.SHIFT_RIGHT:
                result = int(left) >> int(right)
            elif op == _classical_expr.Binary.Op.ADD:
                result = left + right
            elif op == _classical_expr.Binary.Op.SUB:
                result = left - right
            elif op == _classical_expr.Binary.Op.MUL:
                result = left * right
            elif op == _classical_expr.Binary.Op.DIV:
                try:
                    result = left / right
                except Exception:  # pragma: no cover - defensive divide-by-zero
                    result = 0
            else:  # pragma: no cover - unknown operators default to bool equality
                result = left == right
            return self._cast_value(result, node.type)

        def visit_cast(self, node: "_expr_typing.Cast", /) -> Any:
            value = node.operand.accept(self)
            return self._cast_value(value, node.type)

        def visit_index(self, node: "_expr_typing.Index", /) -> Any:
            index_value = node.index.accept(self)
            try:
                resolved_index = int(index_value)
            except Exception:  # pragma: no cover - defensive fallback
                resolved_index = 0

            target_expr = node.target
            if isinstance(target_expr, _classical_expr.Var):
                var_target = target_expr.var
                if isinstance(var_target, Clbit):
                    return self._cast_value(self._debugger._bit_value(var_target), node.type)
                if isinstance(var_target, ClassicalRegister):
                    bits = list(var_target)
                    if 0 <= resolved_index < len(bits):
                        value = self._debugger._bit_value(bits[resolved_index])
                    else:  # pragma: no cover - out-of-range index
                        value = 0
                    return self._cast_value(value, node.type)

            target_value = target_expr.accept(self)
            value: Any
            try:
                value = target_value[resolved_index]
            except Exception:
                if isinstance(target_value, int):
                    value = (int(target_value) >> resolved_index) & 1
                else:  # pragma: no cover - unhandled types default to 0
                    value = 0
            return self._cast_value(value, node.type)

        def visit_stretch(self, node: "_expr_typing.Stretch", /) -> Any:
            # Stretch variables cannot currently be resolved from debugger state; default to 0.
            return 0

        def _cast_value(self, value: Any, target_type: Any) -> Any:
            if _classical_types is None or target_type is None:
                return value
            try:
                if isinstance(target_type, _classical_types.Bool):
                    return bool(value)
                if isinstance(target_type, _classical_types.Uint):
                    return int(value)
                if isinstance(target_type, _classical_types.Float):
                    return float(value)
            except Exception:  # pragma: no cover - defensive fallback
                return value
            return value

else:  # pragma: no cover - accelerated module missing

    class _ClassicalExprEvaluator:  # type: ignore[override]
        def __init__(self, debugger: "CircuitDebugger") -> None:
            raise RuntimeError("Classical expression support is unavailable in this environment.")


@dataclass
class TraceRecord:
    """Immutable snapshot of the debugger at a point in execution.

    Attributes:
        step_index: 0 for the initial state; 1..N after each operation executes.
        instruction: Executed instruction name; ``None`` for the initial state.
        state: Copy of the current :class:`~qiskit.quantum_info.Statevector`.
        classical_bits: Last-known classical bit values in circuit order
            (``None`` for unknown/unset).
        pre_measurement_state: Optional copy of the state immediately before a
            measurement is applied. ``None`` when the record does not correspond
            to a measurement or when the pre-measurement snapshot is unavailable.
    """

    step_index: int  # 0 for initial state, 1..N after each op
    instruction: Optional[str]  # None for initial
    state: Statevector
    classical_bits: List[Optional[int]]  # last-known classical bit values
    pre_measurement_state: Optional[Statevector] = None  # state prior to a measurement step

    def to_dict(
        self, state_format: str = "probs", *, include_pre_measurement: bool = False
    ) -> Dict[str, Any]:
        """Serialize this snapshot.

        Args:
            state_format: Output format for the quantum state. Supported values:
                - ``"probs"``: include ``probabilities_dict`` as the ``state`` field.
                - ``"amplitudes"``: include complex amplitudes formatted as
                  ``[re, im]`` lists under the ``state`` field.
            include_pre_measurement: When ``True`` and this record corresponds to a
                measurement, also serialize :attr:`pre_measurement_state` using the
                requested ``state_format``.

        Returns:
            dict: A JSON-serializable dictionary representing this record.

        Raises:
            ValueError: If ``state_format`` is not one of the supported values.
        """

        def _encode_state(state: Statevector) -> Any:
            if state_format == "probs":
                if state.num_qubits == 0:
                    # ``Statevector.probabilities_dict`` raises ``ValueError`` when the state has no
                    # qubits because it cannot infer a basis label. Represent the unique basis state
                    # explicitly so consumers still receive a well-defined probability distribution.
                    return {"": 1.0}
                return normalize_probability_dict(
                    state.probabilities_dict(), num_qubits=state.num_qubits
                )
            if state_format == "amplitudes":
                encoded: List[List[float]] = []
                for amplitude in state.data:
                    real_part = float(amplitude.real)
                    imag_part = float(amplitude.imag)
                    if not math.isfinite(real_part) or not math.isfinite(imag_part):
                        raise TypeError(
                            "Statevector contains non-finite amplitude components; "
                            "cannot serialize as amplitudes."
                        )
                    encoded.append([real_part, imag_part])
                return encoded
            raise ValueError("state_format must be 'probs' or 'amplitudes'")

        out: Dict[str, Any] = {
            "step_index": self.step_index,
            "instruction": self.instruction,
            "classical_bits": [None if b is None else int(b) for b in self.classical_bits],
        }
        out["state"] = _encode_state(self.state)
        if include_pre_measurement and self.pre_measurement_state is not None:
            out["pre_measurement_state"] = _encode_state(self.pre_measurement_state)
        return out


def _normalize_parameter_binding(
    parameter_values: Optional[Mapping[Any, Any] | Sequence[Any]],
) -> Optional[Mapping[Any, Any] | Sequence[Any]]:
    """Return ``parameter_values`` flattened when wrapped in a single binding container."""

    if parameter_values is None:
        return None
    if isinstance(parameter_values, Mapping):
        return parameter_values
    if isinstance(parameter_values, Sequence) and not isinstance(parameter_values, (str, bytes)):
        if len(parameter_values) == 1:
            first = parameter_values[0]
            if isinstance(first, Mapping) or (
                isinstance(first, Sequence) and not isinstance(first, (str, bytes))
            ):
                return first
        return parameter_values
    return parameter_values


class CircuitDebugger:
    """Exact stepper for Qiskit 2.0+ circuit semantics.

    Executes instructions one-by-one on a statevector while modeling measurement
    collapse, classical bits, and :class:`~qiskit.circuit.IfElseOp` control flow.
    Useful for walkthrough-style debugging and deterministic trace generation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Statevector | int | str | list | tuple] = None,
        seed: Optional[int] = None,
        condition_evaluator: Optional[
            Callable[[Any, List[Optional[int]], QuantumCircuit], bool]
        ] = None,
        parameter_values: Optional[Mapping[Any, Any] | Sequence[Any]] = None,
    ) -> None:
        self.circuit = circuit.copy()
        normalized_params = _normalize_parameter_binding(parameter_values)
        if normalized_params is not None:
            try:
                self.circuit = self.circuit.assign_parameters(normalized_params, inplace=False)
            except Exception as exc:  # pragma: no cover - defensive: assignment should succeed
                raise TypeError(
                    "Failed to bind parameter values onto the circuit; ensure the mapping/sequence "
                    "matches the circuit parameters."
                ) from exc
        self.num_qubits = self.circuit.num_qubits
        self.instructions = list(self.circuit.data)
        self._initial = self._make_initial(initial_state)
        self.state = self._initial.copy()
        self.classical_bits: List[Optional[int]] = [None] * len(self.circuit.clbits)
        self._ip = 0
        # ``seed`` is accepted for API compatibility. Measurements are deterministic so the seed
        # is currently unused, but retained for forward compatibility should stochastic
        # measurement be made configurable in the future.
        self._seed = seed
        self._cond_eval = condition_evaluator
        # Optional control-flow marker emission
        self._emit_markers = False
        self._pending_markers = []  # type: List[TraceRecord]
        # Optional flattened trace callback
        self._record_callback: Optional[Callable[[TraceRecord], None]] = None
        self._flat_next_step = 0
        # Cached evaluator for accelerated classical expressions
        self._expr_evaluator: Optional[_ClassicalExprEvaluator] = None  # type: ignore[assignment]
        self._last_pre_measurement_state: Optional[Statevector] = None

    def reset(self) -> None:
        """Reset to the initial state and clear classical bits."""
        self.state = self._initial.copy()
        self.classical_bits = [None] * len(self.circuit.clbits)
        self._ip = 0
        self._last_pre_measurement_state = None

    def _get_expr_evaluator(self) -> Optional[_ClassicalExprEvaluator]:
        """Lazily instantiate the accelerated classical expression evaluator."""

        if not _CLASSICAL_EXPR_BASE:
            return None
        if self._expr_evaluator is None:
            self._expr_evaluator = _ClassicalExprEvaluator(self)
        return self._expr_evaluator

    def _evaluate_expr_raw(self, obj: Any) -> Any:
        """Best-effort evaluation of a Qiskit classical expression tree."""

        if _CLASSICAL_EXPR_BASE and isinstance(obj, _CLASSICAL_EXPR_BASE):
            evaluator = self._get_expr_evaluator()
            if evaluator is not None:
                try:
                    return evaluator.evaluate(obj)
                except Exception:  # pragma: no cover - defensive fallback
                    return None
        return None

    def step(self) -> TraceRecord:
        """Execute the next instruction (supports nested ``IfElseOp``).

        Returns:
            TraceRecord: Snapshot after executing the next instruction.

        Raises:
            StopIteration: If all instructions have already executed.
        """
        if self._ip >= len(self.instructions):
            raise StopIteration("All instructions executed.")

        ci = self.instructions[self._ip]
        inst = ci.operation
        qargs = ci.qubits
        cargs = ci.clbits
        name = inst.name

        # control flow
        if isinstance(inst, IfElseOp):
            # reset pending markers for this control-flow instruction
            self._pending_markers = []
            self._execute_ifelse(inst)
            self._ip += 1
            self._last_pre_measurement_state = None
            return TraceRecord(self._ip, "if_else", self.state.copy(), self.classical_bits.copy())

        if isinstance(inst, ForLoopOp):
            self._pending_markers = []
            self._execute_for_loop(inst)
            self._ip += 1
            self._last_pre_measurement_state = None
            return TraceRecord(self._ip, "for_loop", self.state.copy(), self.classical_bits.copy())

        if isinstance(inst, WhileLoopOp):
            self._pending_markers = []
            self._execute_while_loop(inst)
            self._ip += 1
            self._last_pre_measurement_state = None
            return TraceRecord(
                self._ip, "while_loop", self.state.copy(), self.classical_bits.copy()
            )

        if isinstance(inst, SwitchCaseOp):
            self._pending_markers = []
            self._execute_switch_case(inst)
            self._ip += 1
            self._last_pre_measurement_state = None
            return TraceRecord(
                self._ip, "switch_case", self.state.copy(), self.classical_bits.copy()
            )

        # Normal instruction
        self._apply_instruction(inst, qargs, cargs)

        self._ip += 1
        pre_measurement_state = (
            self._last_pre_measurement_state.copy()
            if self._last_pre_measurement_state is not None
            else None
        )
        record = TraceRecord(
            self._ip,
            name,
            self.state.copy(),
            self.classical_bits.copy(),
            pre_measurement_state,
        )
        self._last_pre_measurement_state = None
        return record

    def run_all(self) -> TraceRecord:
        """Execute until completion and return the final snapshot."""
        rec = TraceRecord(0, None, self.state.copy(), self.classical_bits.copy())
        while True:
            try:
                rec = self.step()
            except StopIteration:
                return rec

    def trace(
        self,
        include_initial: bool = True,
        include_markers: bool = False,
        *,
        flatten_control_flow: bool = False,
    ) -> List[TraceRecord]:
        """Return a list of snapshots for the whole program.

        Args:
            include_initial: Include the initial state as the first record.
            include_markers: Emit additional records marking entry/exit of
                control-flow branches (if/else, loops, switch cases).
            flatten_control_flow: When ``True``, emit a record for every
                executed instruction within nested control-flow blocks instead
                of only recording the outer operation.

        Returns:
            list[TraceRecord]: Snapshots in execution order (and optionally the
            initial state and control-flow markers).
        """
        if flatten_control_flow:
            return self._trace_flat(
                include_initial=include_initial, include_markers=include_markers
            )
        self.reset()
        self._emit_markers = include_markers
        self._pending_markers = []
        out: List[TraceRecord] = []
        if include_initial:
            out.append(TraceRecord(0, None, self.state.copy(), self.classical_bits.copy()))
        for _ in range(len(self.instructions)):
            rec = self.step()
            if include_markers and self._pending_markers:
                out.extend(self._pending_markers)
                self._pending_markers = []
            out.append(rec)
        return out

    def trace_as_dicts(
        self,
        include_initial: bool = True,
        state_format: str = "probs",
        include_markers: bool = False,
        *,
        flatten_control_flow: bool = False,
        include_pre_measurement: bool = False,
    ) -> List[dict]:
        """Return ``trace()`` output as plain dicts.

        Args:
            include_initial: Whether to include the initial state record.
            state_format: Format for the quantum state (``"probs"`` or ``"amplitudes"``).
            include_markers: Whether to include control-flow marker records.
            include_pre_measurement: Whether to include the ``pre_measurement_state`` field for
                measurement records when available. The state is serialized using the requested
                ``state_format``.

        Returns:
            list[dict]: JSON-serializable trace records.
        """
        recs = self.trace(
            include_initial=include_initial,
            include_markers=include_markers,
            flatten_control_flow=flatten_control_flow,
        )
        return [
            r.to_dict(
                state_format=state_format,
                include_pre_measurement=include_pre_measurement,
            )
            for r in recs
        ]

    def run_until(
        self,
        predicate,
        include_initial: bool = True,
        max_steps: Optional[int] = None,
    ) -> List[TraceRecord]:
        """Run until ``predicate`` is true or the program ends.

        Args:
            predicate: Callable that receives each :class:`TraceRecord` and
                returns ``True`` to stop iteration.
            include_initial: Include the initial state first.
            max_steps: Optional safety cap on the number of executed steps.

        Returns:
            list[TraceRecord]: The records up to and including the one that
            satisfied the predicate (or the last record if none matched).
        """
        self.reset()
        out: List[TraceRecord] = []
        if include_initial:
            rec0 = TraceRecord(0, None, self.state.copy(), self.classical_bits.copy())
            out.append(rec0)
            if predicate(rec0):
                return out

        steps = 0
        while self._ip < len(self.instructions):
            rec = self.step()
            out.append(rec)
            if predicate(rec):
                break
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return out

    def run_until_op(self, name: str, include_initial: bool = True) -> List[TraceRecord]:
        """Run until the next executed instruction has the given name."""
        return self.run_until(lambda r: r.instruction == name, include_initial=include_initial)

    def run_until_index(self, index: int, include_initial: bool = True) -> List[TraceRecord]:
        """Run until the step index equals the given value (1..N)."""
        return self.run_until(lambda r: r.step_index == index, include_initial=include_initial)

    @staticmethod
    def _format_qubit_count(count: int) -> str:
        label = "qubit" if count == 1 else "qubits"
        return f"{count} {label}"

    def _validate_initial_state_vector(self, state: Statevector) -> Statevector:
        """Ensure ``state`` matches the circuit's qubit dimension."""

        qubits = getattr(state, "num_qubits", None)
        if qubits is None:
            raise ValueError(
                "Initial state dimension "
                f"{len(state)} is not a power of two and cannot initialize "
                f"a circuit with {self._format_qubit_count(self.num_qubits)}."
            )
        if qubits != self.num_qubits:
            raise ValueError(
                "Initial state acts on "
                f"{self._format_qubit_count(qubits)} but the circuit has "
                f"{self._format_qubit_count(self.num_qubits)}."
            )
        return state

    def _make_initial(self, val: Optional[Statevector | int | str | list | tuple]) -> Statevector:
        """Coerce various values to an initial Statevector of correct dimension."""

        dimension = 1 << self.num_qubits if self.num_qubits else 1

        if val is None:
            return self._validate_initial_state_vector(Statevector.from_int(0, dimension))

        if isinstance(val, Statevector):
            return self._validate_initial_state_vector(cast(Statevector, val).copy())

        if isinstance(val, int):
            if val < 0 or val >= dimension:
                raise ValueError(
                    "Initial computational basis index "
                    f"{val} is out of range for {self._format_qubit_count(self.num_qubits)}."
                )
            return Statevector.from_int(val, dimension)

        if isinstance(val, str):
            cleaned = "".join(val.split()).replace("_", "")
            if not cleaned:
                if self.num_qubits == 0:
                    return Statevector.from_int(0, dimension)
                raise ValueError(
                    "Initial state bitstring must specify exactly "
                    f"{self._format_qubit_count(self.num_qubits)}."
                )
            invalid = set(cleaned) - {"0", "1"}
            if invalid:
                raise ValueError(
                    "Initial state bitstring must contain only '0' and '1'; "
                    f"got {sorted(invalid)!r} in '{val}'."
                )
            if self.num_qubits == 0:
                raise ValueError(
                    "Initial state bitstring must be empty for a circuit with 0 qubits; "
                    f"got '{val}'."
                )
            if len(cleaned) != self.num_qubits:
                raise ValueError(
                    "Initial state bitstring '"
                    f"{val}' must describe exactly {self._format_qubit_count(self.num_qubits)}."
                )
            basis_index = int(cleaned, 2)
            if basis_index < 0 or basis_index >= dimension:
                raise ValueError(
                    "Initial state bitstring '"
                    f"{val}' is incompatible with {self._format_qubit_count(self.num_qubits)}."
                )
            return Statevector.from_int(basis_index, dimension)

        try:
            state = Statevector(val)
        except Exception as exc:
            raise TypeError("Initial state must be convertible to a Statevector.") from exc
        return self._validate_initial_state_vector(state)

    @staticmethod
    def _bit_from_outcome(outcome: Any) -> int:
        """Return the measured bit value (0/1) from a Qiskit measurement outcome."""

        try:
            return int(outcome) & 1
        except (TypeError, ValueError):
            text = str(outcome).strip()
            for ch in reversed(text):
                if ch in ("0", "1"):
                    return int(ch)
        return 0

    def _collapse_qubit(self, qubit_index: int, outcome: int) -> None:
        """Project ``self.state`` onto ``outcome`` for ``qubit_index`` and renormalize."""

        if self.num_qubits == 0:
            return
        mask = 1 << qubit_index
        data = np.asarray(self.state.data, dtype=complex).copy()
        indices = np.arange(data.size)
        if outcome:
            data[(indices & mask) == 0] = 0.0
        else:
            data[(indices & mask) != 0] = 0.0
        norm = np.linalg.norm(data)
        if norm == 0:
            raise RuntimeError("Deterministic measurement encountered a zero-probability branch.")
        data /= norm
        self.state = Statevector(data, dims=self.state.dims())

    def _measure_qubits(self, indices: List[int]) -> Dict[int, int]:
        """Measure the specified qubit indices sequentially and return their outcomes."""

        measured: Dict[int, int] = {}
        if not indices:
            return measured

        joint = self.state.probabilities_dict(qargs=indices)
        best_label: Optional[str] = None
        best_prob = float("-inf")
        for raw_label, raw_prob in joint.items():
            label = str(raw_label)
            prob = float(raw_prob)
            if prob > best_prob:
                best_prob = prob
                best_label = label
                continue
            if prob == best_prob and best_label is not None:
                try:
                    current = int(label, 2)
                    previous = int(best_label, 2)
                except ValueError:
                    continue
                if current < previous:
                    best_label = label

        if best_label is None:
            best_label = "0" * len(indices)

        for qi, bit_char in zip(indices, reversed(best_label)):
            if qi in measured:
                continue
            bit_value = 1 if bit_char == "1" else 0
            measured[qi] = bit_value
            self._collapse_qubit(qi, bit_value)
        return measured

    def _eval_condition(self, cond: tuple[ClassicalRegister | Clbit, int]) -> bool:
        """Evaluate a classic Qiskit (register/bit, int) condition against current bits."""
        reg_or_bit, cond_val = cond
        if isinstance(reg_or_bit, Clbit):
            idx = self.circuit.find_bit(reg_or_bit).index
            bit_val = self.classical_bits[idx]
            return int(bit_val or 0) == int(cond_val)
        # ClassicalRegister
        acc = 0
        for i, bit in enumerate(reg_or_bit):
            idx = self.circuit.find_bit(bit).index
            acc |= ((self.classical_bits[idx] or 0) & 1) << i
        return acc == int(cond_val)

    def _execute_block(self, block: QuantumCircuit) -> _LoopSignal:
        """Execute all operations in a nested block (supports nested control flow)."""

        for ci in block.data:
            inst = ci.operation
            if isinstance(inst, IfElseOp):
                signal = self._execute_ifelse(inst)
                if signal is not _LoopSignal.NONE:
                    return signal
                continue
            if isinstance(inst, ForLoopOp):
                signal = self._execute_for_loop(inst)
                if signal is not _LoopSignal.NONE:
                    return signal
                continue
            if isinstance(inst, WhileLoopOp):
                signal = self._execute_while_loop(inst)
                if signal is not _LoopSignal.NONE:
                    return signal
                continue
            if isinstance(inst, SwitchCaseOp):
                signal = self._execute_switch_case(inst)
                if signal is not _LoopSignal.NONE:
                    return signal
                continue
            if isinstance(inst, BreakLoopOp):
                self._record_instruction(inst.name)
                return _LoopSignal.BREAK
            if isinstance(inst, ContinueLoopOp):
                self._record_instruction(inst.name)
                return _LoopSignal.CONTINUE
            self._apply_instruction(inst, ci.qubits, ci.clbits)
        return _LoopSignal.NONE

    def _apply_instruction(self, inst, qargs, cargs) -> None:
        """Apply a single instruction to the current state and classical bits."""
        qidx = [self.circuit.find_bit(q).index for q in qargs]

        cond = getattr(inst, "condition", None)
        if cond is not None:
            if isinstance(cond, tuple):
                should_apply = self._eval_condition(cond)
            else:
                if self._cond_eval is not None:
                    should_apply = bool(self._cond_eval(cond, self.classical_bits, self.circuit))
                else:
                    should_apply = self._eval_condition_object(cond)
            if not should_apply:
                self._record_instruction(inst.name)
                self._last_pre_measurement_state = None
                return

        if inst.name == "measure":
            self._last_pre_measurement_state = None
            if not qidx:
                # No qubits to measure; zero out any targeted classical bits for determinism.
                for clbit in cargs:
                    c_index = self.circuit.find_bit(clbit).index
                    self.classical_bits[c_index] = 0
                self._record_instruction(inst.name)
                return

            pre_measurement_state = self.state.copy()
            self._last_pre_measurement_state = pre_measurement_state
            measured = self._measure_qubits(qidx)
            if cargs:
                for q_index, clbit in zip(qidx, cargs):
                    c_index = self.circuit.find_bit(clbit).index
                    self.classical_bits[c_index] = measured.get(q_index, 0)
                if len(cargs) > len(qidx):
                    for clbit in cargs[len(qidx) :]:
                        c_index = self.circuit.find_bit(clbit).index
                        self.classical_bits[c_index] = 0
            self._record_instruction(inst.name, pre_measurement_state=pre_measurement_state)
            return
        else:
            self._last_pre_measurement_state = None
        if inst.name == "barrier":
            self._record_instruction(inst.name)
            return
        if inst.name == "reset":
            self.state = self.state.reset(qidx)
            self._record_instruction(inst.name)
            return

        base_inst = inst
        if cond is not None and hasattr(inst, "copy"):
            try:
                base_inst = inst.copy()
                base_inst.condition = None
            except Exception:
                base_inst = inst

        try:
            op = Operator(base_inst)
        except QiskitError:
            definition = getattr(base_inst, "definition", None)
            if definition is None:
                raise

            # If the composite definition carries parameters, best-effort bind them using
            # the values stored on the instruction.  Qiskit reuses identical ``Parameter``
            # objects between an instruction and its definition, so the zipped order here
            # matches.  Should the lengths differ (for example due to global phase entries
            # or internal scratch parameters), simply fall back to the unbound definition.
            definition_params = list(getattr(definition, "parameters", []))
            inst_params = list(getattr(base_inst, "params", []))
            definition_to_run = definition
            if definition_params and len(definition_params) == len(inst_params):
                binding = dict(zip(definition_params, inst_params))
                try:
                    definition_to_run = definition.assign_parameters(binding, inplace=False)
                except Exception:
                    # Fall back to the original definition if binding fails – the debugger
                    # will raise during evolution if the parameters were essential.
                    definition_to_run = definition

            self.state = self.state.evolve(definition_to_run, qargs=qidx)
        else:
            self.state = self.state.evolve(op, qargs=qidx)
        self._record_instruction(inst.name)
        return

    def _execute_ifelse(self, op: IfElseOp) -> _LoopSignal:
        """Execute an :class:`IfElseOp` by evaluating its condition and running a block."""
        cond = op.condition
        if isinstance(cond, tuple):
            branch_true = self._eval_condition(cond)
        else:
            # Allow a user-provided evaluator for non-tuple expressions; otherwise use internal logic.
            if self._cond_eval is not None:
                branch_true = bool(self._cond_eval(cond, self.classical_bits, self.circuit))
            else:
                branch_true = self._eval_condition_object(cond)
        if branch_true:
            if self._emit_markers:
                self._maybe_emit_marker("enter_if_then")
            signal = self._execute_block(op.blocks[0])
            if self._emit_markers:
                self._maybe_emit_marker("exit_if_then")
            self._record_instruction("if_else")
            return signal
        elif len(op.blocks) > 1 and op.blocks[1] is not None:
            if self._emit_markers:
                self._maybe_emit_marker("enter_if_else")
            signal = self._execute_block(op.blocks[1])
            if self._emit_markers:
                self._maybe_emit_marker("exit_if_else")
            self._record_instruction("if_else")
            return signal
        self._record_instruction("if_else")
        return _LoopSignal.NONE

    @staticmethod
    def _iter_for_loop_values(indexset) -> List:
        """Return the concrete iteration values for a :class:`ForLoopOp`."""

        if isinstance(indexset, range):
            return list(indexset)
        if isinstance(indexset, int):
            return list(range(indexset))
        if isinstance(indexset, tuple):
            if not indexset:
                return []
            if all(isinstance(v, int) for v in indexset) and len(indexset) <= 3:
                if len(indexset) == 1:
                    return list(range(indexset[0]))
                if len(indexset) == 2:
                    start, stop = indexset
                    return list(range(start, stop))
                start, stop, step = indexset[:3]
                return list(range(start, stop, step))
            return list(indexset)
        if isinstance(indexset, list):
            return list(indexset)
        if isinstance(indexset, str):
            return [indexset]
        try:
            return list(indexset)
        except TypeError:
            return [indexset]

    @staticmethod
    def _loop_value_repr(value: Any) -> str:
        text = repr(value)
        return text.replace("\n", " ")

    def _execute_for_loop(self, op: ForLoopOp) -> _LoopSignal:
        """Execute a :class:`ForLoopOp` by iterating its body over the index set."""

        try:
            indexset, loop_param, _ = op.params
        except ValueError:
            # Older Qiskit builds pack parameters differently; fall back to attributes.
            params = list(getattr(op, "params", []))
            indexset = params[0] if params else []
            loop_param = params[1] if len(params) > 1 else None
        values = self._iter_for_loop_values(indexset)
        body = op.blocks[0] if op.blocks else None
        if body is None:
            self._record_instruction("for_loop")
            return _LoopSignal.NONE
        for idx, value in enumerate(values):
            run_block = body
            if loop_param is not None:
                try:
                    run_block = body.assign_parameters({loop_param: value}, inplace=False)
                except Exception as exc:  # pragma: no cover - defensive failure propagation
                    raise TypeError("Failed to bind for-loop parameter to iteration value") from exc
            if self._emit_markers:
                marker = f"for_iter[{idx}:{self._loop_value_repr(value)}]"
                self._maybe_emit_marker(marker)
            signal = self._execute_block(run_block)
            if signal is _LoopSignal.CONTINUE:
                continue
            if signal is _LoopSignal.BREAK:
                break
            if signal is not _LoopSignal.NONE:
                self._record_instruction("for_loop")
                return signal
        self._record_instruction("for_loop")
        return _LoopSignal.NONE

    _MAX_WHILE_ITERATIONS = 4096

    def _execute_while_loop(self, op: WhileLoopOp) -> _LoopSignal:
        """Execute a :class:`WhileLoopOp` until its condition becomes false."""

        body = op.blocks[0] if op.blocks else None
        if body is None:
            self._record_instruction("while_loop")
            return _LoopSignal.NONE
        iterations = 0
        while True:
            cond = op.condition
            if isinstance(cond, tuple):
                keep_running = self._eval_condition(cond)
            else:
                if self._cond_eval is not None:
                    keep_running = bool(self._cond_eval(cond, self.classical_bits, self.circuit))
                else:
                    keep_running = self._eval_condition_object(cond)
            if not keep_running:
                break
            iterations += 1
            if iterations > self._MAX_WHILE_ITERATIONS:
                raise RuntimeError(
                    "WhileLoopOp exceeded maximum iterations; ensure the loop terminates."
                )
            if self._emit_markers:
                self._maybe_emit_marker(f"while_iter[{iterations - 1}]")
            signal = self._execute_block(body)
            if signal is _LoopSignal.CONTINUE:
                continue
            if signal is _LoopSignal.BREAK:
                break
            if signal is not _LoopSignal.NONE:
                self._record_instruction("while_loop")
                return signal
        self._record_instruction("while_loop")
        return _LoopSignal.NONE

    def _execute_switch_case(self, op: SwitchCaseOp) -> _LoopSignal:
        """Execute the matching branch of a :class:`SwitchCaseOp`."""

        cases = op.cases()
        default_block = cases.get(CASE_DEFAULT)
        target_value = self._evaluate_expr_raw(op.target)
        if target_value is None:
            target_value = self._resolve_value(op.target)
        block = cases.get(target_value)
        if block is None and default_block is not None:
            block = default_block
        if block is None:
            self._record_instruction("switch_case")
            return _LoopSignal.NONE
        if self._emit_markers:
            marker = f"switch_case[{self._loop_value_repr(target_value)}]"
            if block is default_block:
                marker = "switch_case_default"
            self._maybe_emit_marker(marker)
        signal = self._execute_block(block)
        self._record_instruction("switch_case")
        return signal

    def _bit_value(self, bit: Clbit) -> int:
        """Return the integer value (0/1) of a single classical bit."""
        idx = self.circuit.find_bit(bit).index
        return int(self.classical_bits[idx] or 0)

    def _register_value(self, reg: ClassicalRegister) -> int:
        """Return the little-endian integer value of a classical register."""
        acc = 0
        for i, bit in enumerate(reg):
            acc |= (self._bit_value(bit) & 1) << i
        return acc

    def get_register_value(self, reg: ClassicalRegister) -> int:
        """Retrieve a register's integer value (LSB at lowest index)."""
        return self._register_value(reg)

    def _resolve_value(self, obj) -> Any:
        """Best-effort conversion of various Qiskit objects/expressions to a scalar."""

        raw_expr_value = self._evaluate_expr_raw(obj)
        if raw_expr_value is not None:
            return raw_expr_value
        # Scalars
        if isinstance(obj, Real):
            return obj
        # Bits and registers
        if isinstance(obj, Clbit):
            return self._bit_value(obj)
        if isinstance(obj, ClassicalRegister):
            return self._register_value(obj)
        # Classical expression variables wrap existing bits/registers via ``var``
        var_obj = getattr(obj, "var", None)
        if isinstance(var_obj, Clbit):
            return self._bit_value(var_obj)
        if isinstance(var_obj, ClassicalRegister):
            return self._register_value(var_obj)
        # Common patterns
        if hasattr(obj, "value"):
            try:
                value = getattr(obj, "value")
                if isinstance(value, Real):
                    return value
                return float(value)
            except Exception:
                pass
        if hasattr(obj, "bits"):
            try:
                bits = list(getattr(obj, "bits"))
                if all(isinstance(b, Clbit) for b in bits):
                    acc = 0
                    for i, b in enumerate(bits):
                        acc |= (self._bit_value(b) & 1) << i
                    return acc
            except Exception:
                pass
        # Expression nodes
        left = getattr(obj, "lhs", None)
        if left is None:
            left = getattr(obj, "left", None)
        right = getattr(obj, "rhs", None)
        if right is None:
            right = getattr(obj, "right", None)
        if left is not None and right is not None:
            try:
                return self._eval_binary_expr(obj)
            except Exception:
                return 0
        if hasattr(obj, "operand"):
            try:
                return self._eval_unary_expr(obj)
            except Exception:
                return 0
        # Default
        return 0

    @staticmethod
    def _normalize_op_name(op_name, default: str = "") -> str:
        """Return a normalized lowercase name for a classical operator."""

        if op_name is None:
            return default
        name = getattr(op_name, "name", None)
        if isinstance(name, str):
            return name.lower()
        text = str(op_name)
        if "." in text:
            text = text.split(".")[-1]
        return text.lower()

    def _eval_binary_expr(self, expr_obj):
        """Evaluate a binary boolean/integer expression Node best-effort."""

        raw_expr_value = self._evaluate_expr_raw(expr_obj)
        if raw_expr_value is not None:
            return raw_expr_value

        op_name = getattr(expr_obj, "op", None) or getattr(expr_obj, "operator", None)
        lhs = getattr(expr_obj, "lhs", None)
        if lhs is None:
            lhs = getattr(expr_obj, "left", None)
        rhs = getattr(expr_obj, "rhs", None)
        if rhs is None:
            rhs = getattr(expr_obj, "right", None)
        if lhs is None or rhs is None:
            return bool(self._resolve_value(expr_obj))

        a = self._resolve_value(lhs)
        b = self._resolve_value(rhs)
        op = self._normalize_op_name(op_name)

        if op in {"==", "eq", "equals", "equal"}:
            return a == b
        if op in {"!=", "ne", "notequals", "not_equals", "not_equal"}:
            return a != b
        if op in {"and", "logic_and", "&&", "&"}:
            return bool(a) and bool(b)
        if op in {"or", "logic_or", "||", "|"}:
            return bool(a) or bool(b)
        if op in {"xor", "logic_xor"}:
            return bool(a) ^ bool(b)
        if op in {"^"}:
            return bool(a) ^ bool(b)
        if op in {"bit_and"}:
            return a & b
        if op in {"bit_or"}:
            return a | b
        if op in {"bit_xor"}:
            return a ^ b
        if op in {"<", "lt", "less"}:
            return a < b
        if op in {"<=", "le", "less_equal"}:
            return a <= b
        if op in {">", "gt", "greater"}:
            return a > b
        if op in {">=", "ge", "greater_equal"}:
            return a >= b
        if op in {"shift_left", "lshift", "<<"}:
            try:
                return a << b
            except Exception:
                return 0
        if op in {"shift_right", "rshift", ">>"}:
            try:
                return a >> b
            except Exception:
                return 0
        if op in {"add", "+"}:
            return a + b
        if op in {"sub", "-"}:
            return a - b
        if op in {"mul", "*"}:
            return a * b
        if op in {"floordiv", "floor_div", "//"}:
            try:
                return a // b
            except Exception:
                return 0
        if op in {"div", "truediv", "true_div", "/"}:
            try:
                return a / b
            except Exception:
                return 0
        if op in {"mod", "remainder", "%"}:
            try:
                return a % b
            except Exception:
                return 0
        if op in {"pow", "**"}:
            try:
                return pow(a, b)
            except Exception:
                return 0
        # Unknown operator
        return False

    def _eval_unary_expr(self, expr_obj):
        """Evaluate a unary boolean/integer expression Node best-effort."""

        raw_expr_value = self._evaluate_expr_raw(expr_obj)
        if raw_expr_value is not None:
            return raw_expr_value

        op_name = getattr(expr_obj, "op", None) or getattr(expr_obj, "operator", None)
        operand = getattr(expr_obj, "operand", None)
        if operand is None:
            return bool(self._resolve_value(expr_obj))

        # Cast nodes lack an explicit operator; treat them as passthrough conversions.
        if op_name is None and hasattr(expr_obj, "type"):
            return self._resolve_value(operand)

        op = self._normalize_op_name(op_name, default="not")
        raw_val = self._resolve_value(operand)

        if op in {"not", "logic_not", "!"}:
            return not bool(raw_val)
        if op in {"bit_not", "~"}:
            return ~int(raw_val)
        if op in {"neg", "minus"}:
            return -raw_val
        if op in {"pos", "plus"}:
            return +raw_val
        return bool(raw_val)

    def _eval_condition_object(self, cond_obj) -> bool:
        """Evaluate a condition expressed as a Qiskit AST-like object or bit/register."""
        # Direct booleans
        raw_expr_value = self._evaluate_expr_raw(cond_obj)
        if raw_expr_value is not None:
            return bool(raw_expr_value)

        if isinstance(cond_obj, bool):
            return cond_obj
        # Single bit
        if isinstance(cond_obj, Clbit):
            return bool(self._bit_value(cond_obj))
        # Expression nodes
        if hasattr(cond_obj, "var"):
            var_obj = getattr(cond_obj, "var")
            if isinstance(var_obj, Clbit):
                return bool(self._bit_value(var_obj))
            if isinstance(var_obj, ClassicalRegister):
                return bool(self._register_value(var_obj))
        left = getattr(cond_obj, "lhs", None)
        if left is None:
            left = getattr(cond_obj, "left", None)
        right = getattr(cond_obj, "rhs", None)
        if right is None:
            right = getattr(cond_obj, "right", None)
        if left is not None and right is not None:
            return bool(self._eval_binary_expr(cond_obj))
        if hasattr(cond_obj, "operand"):
            return bool(self._eval_unary_expr(cond_obj))
        # Registers or bit-like objects
        if isinstance(cond_obj, ClassicalRegister):
            return self._register_value(cond_obj) != 0
        if hasattr(cond_obj, "bits"):
            try:
                bits = list(getattr(cond_obj, "bits"))
                return any(self._bit_value(b) for b in bits if isinstance(b, Clbit))
            except Exception:
                pass
        # Try value attribute
        if hasattr(cond_obj, "value"):
            try:
                return bool(int(getattr(cond_obj, "value")))
            except Exception:
                pass
        return False

    def _record_instruction(
        self, name: str, *, pre_measurement_state: Optional[Statevector] = None
    ) -> None:
        if self._record_callback is None:
            return
        self._flat_next_step += 1
        record = TraceRecord(
            self._flat_next_step,
            name,
            self.state.copy(),
            self.classical_bits.copy(),
            pre_measurement_state.copy() if pre_measurement_state is not None else None,
        )
        self._record_callback(record)

    def _maybe_emit_marker(self, name: str) -> None:
        if not self._emit_markers:
            return
        if self._record_callback is not None:
            self._record_instruction(name)
            return
        self._pending_markers.append(
            TraceRecord(self._ip, name, self.state.copy(), self.classical_bits.copy())
        )

    def _trace_flat(self, *, include_initial: bool, include_markers: bool) -> List[TraceRecord]:
        """Return a flattened trace that emits each executed instruction."""

        prev_emit_markers = self._emit_markers
        prev_pending = self._pending_markers
        prev_callback = self._record_callback
        prev_flat_next = self._flat_next_step
        try:
            self.reset()
            records: List[TraceRecord] = []
            if include_initial:
                records.append(TraceRecord(0, None, self.state.copy(), self.classical_bits.copy()))
            self._flat_next_step = 0
            self._record_callback = records.append
            self._emit_markers = include_markers
            self._pending_markers = []
            while self._ip < len(self.instructions):
                self.step()
            return records
        finally:
            self._emit_markers = prev_emit_markers
            self._pending_markers = prev_pending
            self._record_callback = prev_callback
            self._flat_next_step = prev_flat_next


class _LoopSignal(Enum):
    """Sentinel enum to propagate ``break`` / ``continue`` out of nested blocks."""

    NONE = auto()
    BREAK = auto()
    CONTINUE = auto()
