"""Prefix helper circuit construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit  # type: ignore[import-untyped]
from qiskit.circuit import ClassicalRegister, Instruction  # type: ignore[import-untyped]

__all__ = ["build_prefix_circuits", "build_prefix_circuits_for_qubits"]


@dataclass(frozen=True)
class _InstructionLayout:
    """Lightweight view of a circuit instruction with qubit/clbit indices."""

    operation: Instruction
    qubit_indices: Tuple[int, ...]
    clbit_indices: Tuple[int, ...]
    block_qubit_indices: Tuple[int, ...] = ()
    block_written_clbits: Tuple[int, ...] = ()


class MeasurementTracker:
    """Track which qubits have fresh measurement results."""

    def __init__(self, num_qubits: int, num_clbits: int) -> None:
        self._measured: List[bool] = [False] * num_qubits
        self._dirty: List[bool] = [False] * num_qubits
        self._classical_sources: List[Optional[int]] = [None] * num_clbits

    @classmethod
    def for_sizes(cls, num_qubits: int, num_clbits: int) -> "MeasurementTracker":
        return cls(num_qubits, num_clbits)

    def copy(self) -> "MeasurementTracker":
        clone = MeasurementTracker(len(self._measured), len(self._classical_sources))
        clone._measured = self._measured.copy()
        clone._dirty = self._dirty.copy()
        clone._classical_sources = self._classical_sources.copy()
        return clone

    def _invalidate_qubits(self, qubits: Iterable[int]) -> None:
        for qi in qubits:
            if 0 <= qi < len(self._measured):
                self._measured[qi] = False
                self._dirty[qi] = True

    def _invalidate_classical_bit(self, idx: int) -> None:
        if 0 <= idx < len(self._classical_sources):
            prev_q = self._classical_sources[idx]
            self._classical_sources[idx] = None
            if prev_q is not None and 0 <= prev_q < len(self._measured):
                self._measured[prev_q] = False
                self._dirty[prev_q] = True

    def note_operation(
        self,
        operation: Instruction,
        qubit_indices: Sequence[int],
        clbit_indices: Sequence[int],
        block_qubit_indices: Sequence[int] = (),
        block_written_clbits: Sequence[int] = (),
    ) -> None:
        """Update bookkeeping for an executed operation."""

        name = getattr(operation, "name", "")
        has_condition = getattr(operation, "condition", None) is not None

        if block_written_clbits:
            for ci in block_written_clbits:
                self._invalidate_classical_bit(ci)

        if block_qubit_indices:
            self._invalidate_qubits(block_qubit_indices)

        if name == "measure":
            if has_condition:
                # Conditional measurements might not execute at runtime. Treat the
                # target qubits as unmeasured and clear any classical targets so the
                # previous qubits stored there are remeasured later.
                self._invalidate_qubits(qubit_indices)
                for ci in clbit_indices:
                    self._invalidate_classical_bit(ci)
                return

            for qi in qubit_indices:
                if 0 <= qi < len(self._measured):
                    self._measured[qi] = True
                    self._dirty[qi] = False

            for qi, ci in zip(qubit_indices, clbit_indices):
                if ci < 0 or ci >= len(self._classical_sources):
                    continue
                prev_q = self._classical_sources[ci]
                self._classical_sources[ci] = qi
                if prev_q is not None and prev_q != qi and 0 <= prev_q < len(self._measured):
                    self._measured[prev_q] = False
                    self._dirty[prev_q] = True
            return

        if clbit_indices:
            for ci in clbit_indices:
                self._invalidate_classical_bit(ci)

        if not qubit_indices:
            return

        if name == "barrier":
            return

        for qi in qubit_indices:
            if 0 <= qi < len(self._measured) and self._measured[qi]:
                self._dirty[qi] = True

    def qubits_needing_measurement(self, selection: Optional[Sequence[int]] = None) -> List[int]:
        """Return qubits that must be measured to obtain up-to-date values."""

        if selection is None:
            iterator: Iterable[int] = range(len(self._measured))
        else:
            iterator = selection

        pending: List[int] = []
        for qi in iterator:
            if qi < 0 or qi >= len(self._measured):
                raise ValueError(
                    f"Requested qubit index {qi} is out of range for a circuit with "
                    f"{len(self._measured)} qubits."
                )
            if not self._measured[qi] or self._dirty[qi]:
                pending.append(qi)
        return pending

    def mark_measured(self, qubit: int) -> None:
        if 0 <= qubit < len(self._measured):
            self._measured[qubit] = True
            self._dirty[qubit] = False


class _PrefixBuilder:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._layouts: List[_InstructionLayout] = _compute_instruction_layouts(circuit)
        self._num_qubits = circuit.num_qubits
        self._num_clbits = circuit.num_clbits

    def _fresh_tracker(self) -> MeasurementTracker:
        return MeasurementTracker.for_sizes(self._num_qubits, self._num_clbits)

    def _append_instruction(
        self,
        prefix: QuantumCircuit,
        tracker: MeasurementTracker,
        layout: _InstructionLayout,
    ) -> None:
        qargs = [prefix.qubits[i] for i in layout.qubit_indices]
        cargs = [prefix.clbits[i] for i in layout.clbit_indices]
        prefix.append(layout.operation, qargs, cargs)
        tracker.note_operation(
            layout.operation,
            layout.qubit_indices,
            layout.clbit_indices,
            block_qubit_indices=layout.block_qubit_indices,
            block_written_clbits=layout.block_written_clbits,
        )

    def _finalize_prefix(
        self,
        index: int,
        selection: Optional[Sequence[int]],
        scratch_name: str,
    ) -> QuantumCircuit:
        prefix = QuantumCircuit(self._num_qubits, self._num_clbits, name=f"prefix_{index}")
        tracker = self._fresh_tracker()
        for layout in self._layouts[:index]:
            self._append_instruction(prefix, tracker, layout)

        pending = tracker.qubits_needing_measurement(selection)
        if pending:
            reg = ClassicalRegister(len(pending), scratch_name)
            prefix.add_register(reg)
            for offset, qubit in enumerate(pending):
                prefix.measure(prefix.qubits[qubit], reg[offset])
                tracker.mark_measured(qubit)
        return prefix

    def build_all(self) -> List[QuantumCircuit]:
        return [self._finalize_prefix(i, None, "extra_m") for i in range(1, len(self._layouts) + 1)]

    def build_for_qubits(self, qubits: Sequence[int]) -> List[QuantumCircuit]:
        selection = list(qubits)
        return [
            self._finalize_prefix(i, selection, "marg_m") for i in range(1, len(self._layouts) + 1)
        ]


def _collect_control_flow_effects(
    circuit: QuantumCircuit, operation: Instruction
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return qubit/clbit indices touched by nested control-flow blocks."""

    blocks = getattr(operation, "blocks", None)
    if not blocks:
        return (), ()

    touched_qubits: set[int] = set()
    written_clbits: set[int] = set()

    for block in blocks:
        if block is None:
            continue
        for inner in block.data:
            # Map qubits/clbits through the parent circuit for stable indices.
            q_indices = tuple(circuit.find_bit(q).index for q in inner.qubits)
            touched_qubits.update(q_indices)

            c_indices = tuple(circuit.find_bit(c).index for c in inner.clbits)
            written_clbits.update(c_indices)

            nested_qubits, nested_clbits = _collect_control_flow_effects(circuit, inner.operation)
            touched_qubits.update(nested_qubits)
            written_clbits.update(nested_clbits)

    return tuple(sorted(touched_qubits)), tuple(sorted(written_clbits))


def _compute_instruction_layouts(circuit: QuantumCircuit) -> List[_InstructionLayout]:
    layouts: List[_InstructionLayout] = []
    for instruction in circuit.data:
        q_indices = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        c_indices = tuple(circuit.find_bit(c).index for c in instruction.clbits)
        block_qubits, block_clbits = _collect_control_flow_effects(circuit, instruction.operation)
        layouts.append(
            _InstructionLayout(
                instruction.operation,
                q_indices,
                c_indices,
                block_qubit_indices=block_qubits,
                block_written_clbits=block_clbits,
            )
        )
    return layouts


def build_prefix_circuits(circuit: QuantumCircuit) -> List[QuantumCircuit]:
    """Return helper circuits that end with measurements for every dirty qubit."""

    builder = _PrefixBuilder(circuit)
    return builder.build_all()


def build_prefix_circuits_for_qubits(
    circuit: QuantumCircuit, qubits: Sequence[int]
) -> List[QuantumCircuit]:
    """Return helper circuits that only measure selected qubits when needed."""

    builder = _PrefixBuilder(circuit)
    return builder.build_for_qubits(qubits)
