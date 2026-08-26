# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin facade for recipes DSA offload operators.

The DSA kernels are compiled and installed from ``cann-recipes-infer``. This
module owns only the stable Python call contract and PA validation; it does
not implement a fallback gather kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DSA_BLOCK_SIZE = 128
DSA_TOPK = 2048
DSA_RAW_SEQ = frozenset({1, 4})


@dataclass(frozen=True)
class DsaOffloadConfig:
    """Static DSA bucket attributes used by eager and ACL-graph paths."""

    raw_seq: int = 1
    topk: int = DSA_TOPK
    selection_block_size: int = DSA_BLOCK_SIZE
    compact_layout: int = 1

    def __post_init__(self) -> None:
        if self.raw_seq not in DSA_RAW_SEQ:
            raise ValueError(f"raw_seq must be one of {sorted(DSA_RAW_SEQ)}, got {self.raw_seq}")
        if self.topk != DSA_TOPK:
            raise ValueError(f"The first release supports topk={DSA_TOPK}, got {self.topk}")
        if self.selection_block_size != DSA_BLOCK_SIZE:
            raise ValueError(
                f"The first release supports selection_block_size={DSA_BLOCK_SIZE}, "
                f"got {self.selection_block_size}"
            )
        if self.compact_layout not in (0, 1):
            raise ValueError("compact_layout must be 0 or 1")


def _load_custom_ops_package() -> None:
    """Import the independently built recipes operator package.

    Importing ``custom_ops`` loads its compiled extension and TorchAir
    converters. vLLM Ascend deliberately does not register, redefine, or
    provide a fallback implementation for any DSA operator.
    """

    try:
        import custom_ops  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "The recipes custom_ops package is not installed; build and install "
            "it from cann-recipes-infer/ops/ascendc/torch_ops_extension first"
        ) from error


def _custom_op(name: str) -> Any:
    _load_custom_ops_package()
    custom = getattr(torch.ops, "custom", None)
    op = getattr(custom, name, None) if custom is not None else None
    if op is None:
        raise RuntimeError(
            f"torch.ops.custom.{name} is unavailable; install the recipes DSA extension "
            "built from cann-recipes-infer/ops/ascendc"
        )
    return op


def _require_tensor(name: str, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")


def validate_full_kv_block_table(
    full_kv_block_table: torch.Tensor,
    *,
    full_kv_cache: torch.Tensor,
    full_kv_actual_seq: torch.Tensor | None = None,
    block_size: int = DSA_BLOCK_SIZE,
) -> None:
    """Validate scheduler PA table without synchronising device tensors."""

    _require_tensor("full_kv_block_table", full_kv_block_table, dtype=torch.int32)
    if full_kv_block_table.ndim != 2:
        raise ValueError("full_kv_block_table must be [batch, max_blocks_per_seq]")
    if full_kv_cache.ndim < 2 or full_kv_cache.shape[1] != block_size:
        raise ValueError(
            f"full_kv_cache must use [physical_blocks, {block_size}, ...] layout, "
            f"got {tuple(full_kv_cache.shape)}"
        )
    if full_kv_block_table.shape[0] == 0:
        raise ValueError("full_kv_block_table batch must be non-empty")
    if full_kv_actual_seq is not None:
        _require_tensor("full_kv_actual_seq", full_kv_actual_seq)
        if full_kv_actual_seq.ndim != 1 or full_kv_actual_seq.shape[0] != full_kv_block_table.shape[0]:
            raise ValueError("full_kv_actual_seq must have one entry per block-table row")
    # ``-1`` is the only invalid sentinel accepted by recipes. Avoid a
    # device-side min/max check here because it would synchronize the hot path.
    if full_kv_block_table.device.type == "cpu":
        values = full_kv_block_table
        if bool((values < -1).any()):
            raise ValueError("full_kv_block_table contains an invalid sentinel below -1")
        if bool((values >= full_kv_cache.shape[0]).any()):
            raise ValueError("full_kv_block_table contains a physical block outside full_kv_cache")


def validate_selection_inputs(
    selection_topk_indices: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    *,
    config: DsaOffloadConfig,
) -> None:
    _require_tensor(
        "selection_topk_indices", selection_topk_indices, dtype=torch.int32
    )
    _require_tensor("full_kv_actual_seq", full_kv_actual_seq, dtype=torch.int32)
    expected_ndim = 3 if config.raw_seq == 1 else 4
    if selection_topk_indices.ndim != expected_ndim:
        raise ValueError(
            "selection_topk_indices must use the recipes DsaPlan layout: "
            "[batch, 1, topk] for raw_seq=1 or "
            "[batch, raw_seq, 1, topk] for raw_seq=4; "
            f"got ndim={selection_topk_indices.ndim} for raw_seq={config.raw_seq}"
        )
    if selection_topk_indices.shape[-1] != config.topk:
        raise ValueError(
            f"selection_topk_indices last dimension must be {config.topk}, "
            f"got {selection_topk_indices.shape[-1]}"
        )
    if config.raw_seq == 1:
        if selection_topk_indices.shape[1] != 1:
            raise ValueError(
                "raw_seq=1 requires selection_topk_indices shape "
                "[batch, 1, topk]"
            )
    elif selection_topk_indices.shape[1:3] != (config.raw_seq, 1):
        raise ValueError(
            "raw_seq=4 requires selection_topk_indices shape "
            "[batch, raw_seq, 1, topk]"
        )
    batch = selection_topk_indices.shape[0]
    if full_kv_actual_seq.shape != (batch,):
        raise ValueError(
            "full_kv_actual_seq must be [batch] and match "
            f"selection_topk_indices; got {tuple(full_kv_actual_seq.shape)} "
            f"for batch={batch}"
        )


def dsa_plan(
    selection_topk_indices: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    pool_ids: torch.Tensor,
    id_to_slot: torch.Tensor,
    lru_counter: torch.Tensor,
    *,
    config: DsaOffloadConfig | None = None,
    group_id: int = 0,
    owner_layer: int = 0,
    group_kind: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config = config or DsaOffloadConfig()
    validate_selection_inputs(selection_topk_indices, full_kv_actual_seq, config=config)
    for name, value in (("pool_ids", pool_ids), ("id_to_slot", id_to_slot), ("lru_counter", lru_counter)):
        _require_tensor(name, value)
    result = _custom_op("dsa_plan")(
        selection_topk_indices,
        full_kv_actual_seq,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=config.raw_seq,
        group_id=group_id,
        owner_layer=owner_layer,
        group_kind=group_kind,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("dsa_plan must return (plan, install_records, selection_kv_actual_seq)")
    return result


def dsa_serve(
    plan: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_k_rope: torch.Tensor,
    pool_kv_cache: torch.Tensor,
    pool_k_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    *,
    full_kv_block_table: torch.Tensor | None = None,
    full_kv_actual_seq: torch.Tensor | None = None,
    config: DsaOffloadConfig | None = None,
) -> object | None:
    config = config or DsaOffloadConfig()
    if full_kv_block_table is None:
        raise ValueError("full_kv_block_table is required for layerwise Host KV offload")
    validate_full_kv_block_table(
        full_kv_block_table,
        full_kv_cache=full_kv_cache,
        full_kv_actual_seq=full_kv_actual_seq,
        block_size=config.selection_block_size,
    )
    op = _custom_op("dsa_serve")
    try:
        # PA-aware recipes schema: the table is carried with the full cache,
        # before resident pool and selection output tensors.
        op(
            plan,
            full_kv_cache,
            full_k_rope,
            full_kv_block_table,
            pool_kv_cache,
            pool_k_rope,
            selection_kv_cache,
            selection_k_rope,
            raw_seq=config.raw_seq,
            topk=config.topk,
            selection_block_size=config.selection_block_size,
            compact_layout=config.compact_layout,
        )
    except TypeError as error:
        raise RuntimeError(
            "Installed dsa_serve does not expose the PA full_kv_block_table input; "
            "rebuild recipes DSA ops with the PA schema before enabling Host KV offload"
        ) from error
    npu = getattr(torch, "npu", None)
    event_cls = getattr(npu, "Event", None) if npu is not None else None
    if event_cls is None:
        return None
    event = event_cls()
    event.record()
    return event


def dsa_install(
    install_records: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    pool_kv_cache: torch.Tensor,
    pool_k_rope: torch.Tensor,
    pool_ids: torch.Tensor,
    id_to_slot: torch.Tensor,
    lru_counter: torch.Tensor,
    *,
    config: DsaOffloadConfig | None = None,
    metadata_update: int = 1,
) -> object | None:
    config = config or DsaOffloadConfig()
    _require_tensor("selection_kv_block_table", selection_kv_block_table, dtype=torch.int32)
    if selection_kv_block_table.ndim != 2:
        raise ValueError("selection_kv_block_table must be [batch, max_selection_blocks]")
    if metadata_update not in (0, 1):
        raise ValueError("metadata_update must be 0 or 1")
    _custom_op("dsa_install")(
        install_records,
        selection_kv_cache,
        selection_k_rope,
        selection_kv_block_table,
        pool_kv_cache,
        pool_k_rope,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=config.raw_seq,
        topk=config.topk,
        selection_block_size=config.selection_block_size,
        metadata_update=metadata_update,
    )
    npu = getattr(torch, "npu", None)
    event_cls = getattr(npu, "Event", None) if npu is not None else None
    if event_cls is None:
        return None
    event = event_cls()
    event.record()
    return event
