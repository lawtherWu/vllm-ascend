# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-address resident/selection state for DSA Decode offload.

The scheduler and vLLM ``InputBatch`` remain the owners of request and block
allocation.  This module only tracks the fixed per-layer DSA workspaces and
the request identity currently occupying each resident row.  It deliberately
does not implement an eviction policy or a second request allocator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ResidentOwner:
    request_id: str
    request_generation: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.request_generation < 0:
            raise ValueError("request_generation must be non-negative")


@dataclass(frozen=True)
class ResidentRow:
    row: int
    owner: ResidentOwner | None
    valid: bool


@dataclass(frozen=True)
class DsaLayerWorkspace:
    """Fixed tensors and completion event for one non-MTP sparse layer."""

    layer_id: int
    resident_kv_cache: torch.Tensor
    resident_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_k_rope: torch.Tensor
    selection_block_table: torch.Tensor
    selection_query_lens: torch.Tensor | None = None
    selection_default_indices: torch.Tensor | None = None
    install_event: object | None = None

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.selection_block_table.dtype != torch.int32 or self.selection_block_table.ndim != 2:
            raise ValueError("selection_block_table must be a 2-D int32 tensor")
        rows = self.selection_block_table.shape[0]
        if self.selection_query_lens is not None:
            if self.selection_query_lens.dtype != torch.int32 or self.selection_query_lens.shape != (rows,):
                raise ValueError("selection_query_lens must be a row-sized int32 tensor")
        if self.selection_default_indices is not None:
            if self.selection_default_indices.dtype != torch.int32 or self.selection_default_indices.shape[0] != rows:
                raise ValueError("selection_default_indices must have one int32 row per query")
        if self.resident_kv_cache.shape[0] != self.resident_k_rope.shape[0]:
            raise ValueError("resident KV and RoPE must have the same row capacity")

    @property
    def resident_rows(self) -> int:
        return self.resident_kv_cache.shape[0]


class DsaResidentState:
    """Per-layer row ownership and install ordering without allocation logic."""

    def __init__(self, workspaces: Sequence[DsaLayerWorkspace]):
        self._workspaces = {workspace.layer_id: workspace for workspace in workspaces}
        if len(self._workspaces) != len(workspaces):
            raise ValueError("Duplicate DSA layer workspace")
        self._rows: dict[int, list[ResidentOwner | None]] = {
            layer_id: [None] * workspace.resident_rows
            for layer_id, workspace in self._workspaces.items()
        }
        self._valid: dict[int, list[bool]] = {
            layer_id: [False] * workspace.resident_rows
            for layer_id, workspace in self._workspaces.items()
        }
        self._pending_invalidations: dict[int, set[int]] = {
            layer_id: set() for layer_id in self._workspaces
        }
        self._last_install_events: list[object] = []
        self._lock = threading.RLock()

    @property
    def workspaces(self) -> Mapping[int, DsaLayerWorkspace]:
        return self._workspaces

    def workspace(self, layer_id: int) -> DsaLayerWorkspace:
        try:
            return self._workspaces[layer_id]
        except KeyError as error:
            raise KeyError(f"Unknown non-MTP DSA layer {layer_id}") from error

    def wait_previous_install(self) -> None:
        """Wait for the final install of the previous step before row reuse."""

        for event in tuple(self._last_install_events):
            wait = getattr(event, "wait", None)
            if callable(wait):
                wait()
                continue
            synchronize = getattr(event, "synchronize", None)
            if callable(synchronize):
                synchronize()
                continue
            raise TypeError("DSA install event must provide wait() or synchronize()")

    def begin_step(self) -> None:
        """Make pending row invalidations visible before the next DsaPlan."""

        self.wait_previous_install()
        with self._lock:
            # The events are only needed to protect the transition into this
            # step. Drop them after the barrier so they do not accumulate.
            self._last_install_events.clear()
            for layer_id, rows in self._pending_invalidations.items():
                valid = self._valid[layer_id]
                owners = self._rows[layer_id]
                for row in rows:
                    valid[row] = False
                    owners[row] = None
                rows.clear()

    def invalidate_rows(self, rows: Mapping[int, Sequence[int]]) -> None:
        """Queue row invalidation after InputBatch condense/reorder."""

        with self._lock:
            for layer_id, layer_rows in rows.items():
                workspace = self.workspace(layer_id)
                pending = self._pending_invalidations[layer_id]
                for row in layer_rows:
                    if row < 0 or row >= workspace.resident_rows:
                        raise IndexError(f"Resident row {row} is outside layer {layer_id}")
                    pending.add(row)

    def bind_row(self, layer_id: int, row: int, owner: ResidentOwner) -> None:
        with self._lock:
            workspace = self.workspace(layer_id)
            if row < 0 or row >= workspace.resident_rows:
                raise IndexError(f"Resident row {row} is outside layer {layer_id}")
            self._rows[layer_id][row] = owner
            self._valid[layer_id][row] = True
            self._pending_invalidations[layer_id].discard(row)

    def mark_row_invalid(self, layer_id: int, row: int) -> None:
        self.invalidate_rows({layer_id: (row,)})

    def row_state(self, layer_id: int, row: int) -> ResidentRow:
        with self._lock:
            workspace = self.workspace(layer_id)
            if row < 0 or row >= workspace.resident_rows:
                raise IndexError(f"Resident row {row} is outside layer {layer_id}")
            return ResidentRow(row, self._rows[layer_id][row], self._valid[layer_id][row])

    def rows_for_owner(self, owner: ResidentOwner, layer_id: int) -> tuple[int, ...]:
        with self._lock:
            return tuple(
                row
                for row, current in enumerate(self._rows[layer_id])
                if current == owner and self._valid[layer_id][row]
            )

    def record_final_install_event(self, event: object | None) -> None:
        """Record one group's final install event for the next-step barrier."""

        if event is not None:
            self._last_install_events.append(event)

    def clear(self) -> None:
        with self._lock:
            for layer_id in self._workspaces:
                self._rows[layer_id] = [None] * self._workspaces[layer_id].resident_rows
                self._valid[layer_id] = [False] * self._workspaces[layer_id].resident_rows
                self._pending_invalidations[layer_id].clear()
            self._last_install_events.clear()
