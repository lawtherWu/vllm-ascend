# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
<<<<<<< HEAD
    PAD_SLOT_ID,
=======
    compute_causal_conv1d_metadata,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ops.triton.gdn_chunk_meta import (
    _build_seq_lens,
    _validate_cu_seqlens,
    build_chunk_meta_device,
)

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
# Accepted-token counts are one-based. The zero-length query segment makes
# graph-padding rows no-ops, while 1 keeps their count valid for GDN kernels.
_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        # Boolean mask values are exactly representable in float32. Integer
        # argsort falls back to AiCPU on Ascend; keep this stable partition on
        # AiCore without changing the ordering of tokens within either group.
        tensor = tensor.to(torch.float32)
    return torch.argsort(tensor, stable=True)


<<<<<<< HEAD
def _remove_spec_graph_padding_queries(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> CommonAttentionMetadata:
    """Remove inactive FIA graph rows from the GDN-local query view.

    FIA represents every padded graph request with a full K+1 query span.
    Those rows must be zero-length for the target's recurrent GDN state update.
    The shared MLA metadata remains unchanged.
    """
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    if seq_lens is None or num_decode_draft_tokens_cpu is None:
        return common_attn_metadata
    num_reqs = common_attn_metadata.num_reqs
    draft_tokens = num_decode_draft_tokens_cpu[:num_reqs]
    if draft_tokens.numel() != num_reqs or not torch.any(draft_tokens > 0).item():
        return common_attn_metadata
    inactive = (seq_lens[:num_reqs] == 0) & (draft_tokens < 0)
    if not torch.any(inactive).item():
        return common_attn_metadata
    first_inactive = int(torch.nonzero(inactive, as_tuple=True)[0][0])
    if not torch.all(inactive[first_inactive:]).item():
        raise ValueError("GDN speculative graph padding must be a request suffix")
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.clone()
    query_start_loc_cpu[first_inactive + 1 :] = query_start_loc_cpu[first_inactive]
    return common_attn_metadata.replace(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(common_attn_metadata.query_start_loc.device),
        num_actual_tokens=int(query_start_loc_cpu[-1]),
    )


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode. Once recurrent state exists, both paths must construct
    identical GDN metadata so the graph consumes the current state indices.
    First-token prefills remain on the prefill path because they have no state
    to update.
    """
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if is_prefilling is None or seq_lens_cpu is None:
        return common_attn_metadata

    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    prefill_to_decode = is_prefilling & (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    if not torch.any(prefill_to_decode).item():
        return common_attn_metadata

    is_prefilling = is_prefilling.clone()
    is_prefilling[prefill_to_decode] = False
    return common_attn_metadata.replace(is_prefilling=is_prefilling)


=======
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_cpu: torch.Tensor
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
<<<<<<< HEAD
    chunk_indices_chunk64: torch.Tensor | None
    chunk_offsets_chunk64: torch.Tensor | None
    update_chunk_offsets_chunk64: torch.Tensor | None
    final_chunk_indices_chunk64: torch.Tensor | None
    chunk_indices_large_block: torch.Tensor | None
    block_indices_cumsum: torch.Tensor | None
    num_decodes: int = 0
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None
=======
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    num_decodes: int = 0
    _buffer_slot: object | None = None
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


@dataclass
class GDNCausalConv1dHostMetadata:
    query_start_loc_cpu: torch.Tensor
    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNSpecCausalConv1dHostMetadata:
    query_start_loc_cpu: torch.Tensor
    cache_indices_cpu: torch.Tensor
    num_accepted_tokens_cpu: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNPrefillFallbackMeta:
    causal_conv1d: GDNCausalConv1dHostMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class GDNDecodeFallbackMeta:
    causal_conv1d: GDNCausalConv1dHostMetadata


@dataclass
<<<<<<< HEAD
class GDNSpecDecodeMetadata:
    spec_causal_conv1d: GDNSpecCausalConv1dMetadata
    # 310P derives lengths in its recurrent wrapper and only consumes conv metadata.
    actual_seq_lengths: torch.Tensor | None
=======
class GDNSpecDecodeFallbackMeta:
    spec_causal_conv1d: GDNSpecCausalConv1dHostMetadata
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor


@dataclass
class _GDNCausalConv1dHostBufferSlot:
    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor


@dataclass
class _GDNSpecCausalConv1dHostBufferSlot:
    cache_indices_cpu: torch.Tensor
    num_accepted_tokens_cpu: torch.Tensor


@dataclass
class _GDNChunkMetaSizeInfo:
    num_seqs: int
    num_chunk_indices_chunk64: int
    num_chunk_indices_large_block: int
    num_block_indices_cumsum: int


@dataclass
class _GDNChunkMetaShapeInfo(_GDNChunkMetaSizeInfo):
    chunk_counts_chunk64: torch.Tensor
    chunk_counts_large_block: torch.Tensor
    chunk_counts_cumsum: torch.Tensor


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prepare_chunk_counts_cpu(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _fill_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    cursor = 0
    compact_seq_idx = 0
    for num_chunks in chunk_counts.tolist():
        if num_chunks <= 0:
            continue
        # `prepare_chunk_indices` compacts away zero-length sequences, so the
        # sequence index here must follow the same compact numbering.
        out[cursor : cursor + num_chunks, 0].fill_(compact_seq_idx)
        out[cursor : cursor + num_chunks, 1] = torch.arange(
            num_chunks,
            dtype=out.dtype,
        )
        cursor += num_chunks
        compact_seq_idx += 1
    return cursor


def _build_chunk_indices_host(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> tuple[int, ...]:
    chunk_counts = _prepare_chunk_counts_cpu(cu_seqlens_cpu, chunk_size)
    num_chunk_indices = int(chunk_counts.sum().item())
    chunk_indices = torch.empty((num_chunk_indices, 2), dtype=torch.int64)
    _fill_chunk_indices_cpu(chunk_indices, chunk_counts)
    return tuple(chunk_indices.reshape(-1).tolist())


def _fill_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=out[1 : chunk_counts.numel() + 1],
        )
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[: chunk_counts.numel()])
        out[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


def _build_chunk_meta_shape_info(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkMetaShapeInfo:
    chunk_counts_chunk64 = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_chunk_size,
    )
    chunk_counts_large_block = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_large_block_size,
    )
    chunk_counts_cumsum = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_cumsum_block_size,
    )
    return _GDNChunkMetaShapeInfo(
        num_seqs=chunk_counts_chunk64.numel(),
        num_chunk_indices_chunk64=int(chunk_counts_chunk64.sum().item()),
        num_chunk_indices_large_block=int(chunk_counts_large_block.sum().item()),
        num_block_indices_cumsum=int(chunk_counts_cumsum.sum().item()),
        chunk_counts_chunk64=chunk_counts_chunk64,
        chunk_counts_large_block=chunk_counts_large_block,
        chunk_counts_cumsum=chunk_counts_cumsum,
    )


<<<<<<< HEAD
def _materialize_graph_request_tensor(
    buffer: torch.Tensor,
    values: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
    padding_value: int | bool,
) -> torch.Tensor:
    """Write request-level values into a stable full-graph buffer."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count > buffer.size(0):
        raise ValueError(f"graph_request_count ({graph_request_count}) exceeds buffer capacity ({buffer.size(0)})")
    if actual_request_count > values.size(0):
        raise ValueError(f"actual_request_count ({actual_request_count}) exceeds source rows ({values.size(0)})")

    materialized = buffer[:graph_request_count]
    materialized[:actual_request_count].copy_(
        values[:actual_request_count],
        non_blocking=True,
    )
    materialized[actual_request_count:].fill_(padding_value)
    return materialized


def _materialize_graph_query_start_loc(
    buffer: torch.Tensor,
    query_start_loc: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
) -> torch.Tensor:
    """Pad request boundaries with zero-length dummy request segments."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count + 1 > buffer.size(0):
        raise ValueError(
            f"graph request boundaries ({graph_request_count + 1}) exceed buffer capacity ({buffer.size(0)})"
        )
    if actual_request_count + 1 > query_start_loc.size(0):
        raise ValueError(
            f"actual request boundaries ({actual_request_count + 1}) exceed source rows ({query_start_loc.size(0)})"
        )
    materialized = buffer[: graph_request_count + 1]
    materialized[: actual_request_count + 1].copy_(
        query_start_loc[: actual_request_count + 1],
        non_blocking=True,
    )
    padding = materialized[actual_request_count + 1 :]
    if padding.numel() > 0:
        padding.copy_(
            materialized[actual_request_count : actual_request_count + 1].expand_as(padding),
            non_blocking=True,
        )
    return materialized


def _compact_empty_segments(cu_seqlens_host, initial_state, device=None):
    """Drop zero-length segments so AscendC fwd_h/fwd_o indexing lines up.

    Returns ``(cu_seqlens_kern, initial_state_kern, keep_meta)``:
    cu_seqlens / initial_state with empty segments removed, plus a bool
    mask (None when nothing was removed).  The compacted ``final_state``
    must be scattered back via ``keep_meta`` (empty segments keep their
    initial state).

    When *device* is given, ``keep_meta`` is moved to that device so that
    callers can index NPU tensors without an extra host→device sync.
    """
    if cu_seqlens_host is None:
        return None, initial_state, None
    cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    keep = (cu[1:] - cu[:-1]) > 0
    if bool(keep.all()):
        return cu_seqlens_host, initial_state, None
    # Compute compact cu_seqlens while keep is still on CPU (cu is CPU-only).
    cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
    # Move keep to device only for indexing device-side tensors.
    if device is not None:
        keep = keep.to(device)
    st_kern = initial_state[keep] if initial_state is not None else None
    return cu_kern, st_kern, keep
=======
def _count_chunk_indices_cpu(seq_lens_cpu: torch.Tensor, chunk_size: int) -> int:
    return int(
        torch.div(
            seq_lens_cpu + chunk_size - 1,
            chunk_size,
            rounding_mode="floor",
        )
        .sum()
        .item()
    )
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


def _build_chunk_meta_size_info(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkMetaSizeInfo:
    seq_lens_cpu = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return _GDNChunkMetaSizeInfo(
        num_seqs=seq_lens_cpu.numel(),
        num_chunk_indices_chunk64=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_chunk_size,
        ),
        num_chunk_indices_large_block=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_large_block_size,
        ),
        num_block_indices_cumsum=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_cumsum_block_size,
        ),
    )


def _allocate_chunk_meta_cpu_tensors(shape_info: _GDNChunkMetaSizeInfo) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": torch.empty(
            (shape_info.num_chunk_indices_chunk64, 2),
            dtype=torch.int32,
        ),
        "chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "update_chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "final_chunk_indices_chunk64": torch.empty(
            (shape_info.num_seqs,),
            dtype=torch.int32,
        ),
        "chunk_indices_large_block": torch.empty(
            (shape_info.num_chunk_indices_large_block, 2),
            dtype=torch.int32,
        ),
        "block_indices_cumsum": torch.empty(
            (shape_info.num_block_indices_cumsum, 2),
            dtype=torch.int32,
        ),
    }


def _slice_chunk_meta_slot_tensors(
    slot: _GDNChunkedPrefillBufferSlot,
    shape_info: _GDNChunkMetaSizeInfo,
) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": slot.chunk_indices_chunk64[: shape_info.num_chunk_indices_chunk64],
        "chunk_offsets_chunk64": slot.chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "update_chunk_offsets_chunk64": slot.update_chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "final_chunk_indices_chunk64": slot.final_chunk_indices_chunk64[: shape_info.num_seqs],
        "chunk_indices_large_block": slot.chunk_indices_large_block[: shape_info.num_chunk_indices_large_block],
        "block_indices_cumsum": slot.block_indices_cumsum[: shape_info.num_block_indices_cumsum],
    }


def _fill_chunk_meta_cpu_tensors(
    tensors: dict[str, torch.Tensor],
    shape_info: _GDNChunkMetaShapeInfo,
) -> None:
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_offsets_cpu(
        tensors["chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_update_chunk_offsets_cpu(
        tensors["update_chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_final_chunk_indices_cpu(
        tensors["final_chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_large_block"],
        shape_info.chunk_counts_large_block,
    )
    _fill_chunk_indices_cpu(
        tensors["block_indices_cumsum"],
        shape_info.chunk_counts_cumsum,
    )


def _fill_chunk_meta_device_tensors(
    builder,
    cu_seqlens: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> None:
    seq_lens = None
    validate_inputs = True
    if cu_seqlens.device.type == "npu":
        _validate_cu_seqlens(cu_seqlens, builder._ascend_gdn_chunk_size)
        assert builder._ascend_gdn_large_block_size > 0
        assert builder._ascend_gdn_cumsum_block_size > 0
        seq_lens = _build_seq_lens(cu_seqlens)
        validate_inputs = False
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_chunk_size,
        out_chunk_indices=tensors["chunk_indices_chunk64"],
        out_chunk_offsets=tensors["chunk_offsets_chunk64"],
        out_update_chunk_offsets=tensors["update_chunk_offsets_chunk64"],
        out_final_chunk_indices=tensors["final_chunk_indices_chunk64"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_large_block_size,
        out_chunk_indices=tensors["chunk_indices_large_block"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_cumsum_block_size,
        out_chunk_indices=tensors["block_indices_cumsum"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )


def _build_chunked_prefill_metadata(
    builder,
    tensors: dict[str, torch.Tensor],
    *,
    cu_seqlens_cpu: torch.Tensor,
<<<<<<< HEAD
    device: torch.device,
    build_device_metadata: bool = True,
) -> GDNChunkedPrefillMetadata:
    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    if build_device_metadata:
        hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
        linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
        if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
            gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
        elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
            gdn_num_heads = (
                hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
            )
        else:
            gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(
                builder.vllm_config.parallel_config
            )
        cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
        cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
        chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        chunk_indices_large_block = prepare_chunk_indices(
            cu_seqlens_cpu,
            _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        )
        block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    if build_device_metadata:
        device_chunk_metadata = {
            "chunk_indices_chunk64": chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_offsets_chunk64": chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "update_chunk_offsets_chunk64": update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "final_chunk_indices_chunk64": final_chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_indices_large_block": chunk_indices_large_block.to(device=device, non_blocking=True),
            "block_indices_cumsum": block_indices_cumsum.to(device=device, non_blocking=True),
        }
    else:
        device_chunk_metadata = {
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": None,
            "update_chunk_offsets_chunk64": None,
            "final_chunk_indices_chunk64": None,
            "chunk_indices_large_block": None,
            "block_indices_cumsum": None,
        }

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        **device_chunk_metadata,
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
=======
    slot: _GDNChunkedPrefillBufferSlot | None = None,
) -> GDNChunkedPrefillMetadata:
    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).tolist())
    num_decodes = sum(
        1 for i in range(len(cu_seqlens_host) - 1) if 0 < cu_seqlens_host[i + 1] - cu_seqlens_host[i] <= 1
    )
    return GDNChunkedPrefillMetadata(
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=_build_chunk_indices_host(
            cu_seqlens_cpu,
            builder._ascend_gdn_chunk_size,
        ),
        chunk_indices_chunk64=tensors["chunk_indices_chunk64"],
        chunk_offsets_chunk64=tensors["chunk_offsets_chunk64"],
        update_chunk_offsets_chunk64=tensors["update_chunk_offsets_chunk64"],
        final_chunk_indices_chunk64=tensors["final_chunk_indices_chunk64"],
        chunk_indices_large_block=tensors["chunk_indices_large_block"],
        block_indices_cumsum=tensors["block_indices_cumsum"],
        num_decodes=num_decodes,
        _buffer_slot=slot,
    )


def _get_gdn_num_heads(builder) -> int:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        return hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
    return builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)


def _allocate_chunked_prefill_slot(builder, device: torch.device):
    max_num_batched_tokens = builder.vllm_config.scheduler_config.max_num_batched_tokens
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNChunkedPrefillBufferSlot(
        chunk_indices_chunk64=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        update_chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        final_chunk_indices_chunk64=torch.empty(
            (max_num_seqs,),
            dtype=torch.int32,
            device=device,
        ),
        chunk_indices_large_block=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        block_indices_cumsum=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
    )


def _ensure_chunk_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_chunk_meta_initialized", False):
        return
    builder._ascend_gdn_chunk_meta_initialized = True
    builder._ascend_gdn_chunk_meta_device = device
    builder._ascend_gdn_chunk_size = _GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    gdn_num_heads = _get_gdn_num_heads(builder)
    cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * builder._ascend_gdn_chunk_size))
    builder._ascend_gdn_cumsum_block_size = _next_power_of_2(cumsum_chunks)
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_chunked_prefill_pool = [
            _allocate_chunked_prefill_slot(builder, device),
            _allocate_chunked_prefill_slot(builder, device),
        ]


def _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu: torch.Tensor | None) -> torch.Tensor | None:
    if (
        not getattr(builder, "use_spec_decode", False)
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0].sum().item() == 0
    ):
        return None
    return num_decode_draft_tokens_cpu >= 0


def _build_non_spec_query_start_loc_cpu(
    builder,
    attn_metadata,
    common_attn_metadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> torch.Tensor | None:
    if attn_metadata.num_prefills <= 0 and attn_metadata.num_decodes <= 0:
        return None

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    spec_sequence_masks_cpu = _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu)
    if spec_sequence_masks_cpu is None:
        return query_start_loc_cpu

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
    non_spec_query_start_loc_cpu = torch.zeros(
        non_spec_query_lens_cpu.numel() + 1,
        dtype=query_start_loc_cpu.dtype,
    )
    torch.cumsum(
        non_spec_query_lens_cpu,
        dim=0,
        out=non_spec_query_start_loc_cpu[1:],
    )
    return non_spec_query_start_loc_cpu


def _build_spec_query_start_loc_cpu(
    builder,
    common_attn_metadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> torch.Tensor | None:
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    spec_sequence_masks_cpu = _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu)
    if spec_sequence_masks_cpu is None:
        return query_start_loc_cpu

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
    spec_query_start_loc_cpu = torch.zeros(
        spec_query_lens_cpu.numel() + 1,
        dtype=query_start_loc_cpu.dtype,
    )
    torch.cumsum(
        spec_query_lens_cpu,
        dim=0,
        out=spec_query_start_loc_cpu[1:],
    )
    return spec_query_start_loc_cpu


def _allocate_causal_conv1d_host_slot(
    builder,
    device: torch.device,
) -> _GDNCausalConv1dHostBufferSlot:
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNCausalConv1dHostBufferSlot(
        cache_indices_cpu=torch.empty(
            max_num_seqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
        has_initial_state_cpu=torch.empty(
            max_num_seqs,
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
    )


def _ensure_causal_conv1d_host_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_causal_conv1d_host_meta_initialized", False):
        return
    builder._ascend_gdn_causal_conv1d_host_meta_initialized = True
    builder._ascend_gdn_causal_conv1d_host_pool_idx = -1
    builder._ascend_gdn_causal_conv1d_host_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_causal_conv1d_host_pool = [
            _allocate_causal_conv1d_host_slot(builder, device),
            _allocate_causal_conv1d_host_slot(builder, device),
        ]


def _acquire_causal_conv1d_host_slot(builder) -> _GDNCausalConv1dHostBufferSlot:
    pool = builder._ascend_gdn_causal_conv1d_host_pool
    builder._ascend_gdn_causal_conv1d_host_pool_idx = (builder._ascend_gdn_causal_conv1d_host_pool_idx + 1) % len(pool)
    return pool[builder._ascend_gdn_causal_conv1d_host_pool_idx]


def _allocate_spec_causal_conv1d_host_slot(
    builder,
    device: torch.device,
) -> _GDNSpecCausalConv1dHostBufferSlot:
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    spec_cfg = builder.vllm_config.speculative_config
    num_speculative_tokens = spec_cfg.num_speculative_tokens if spec_cfg else 0
    decode_cudagraph_max_bs = getattr(builder, "decode_cudagraph_max_bs", max_num_seqs)
    max_elements = decode_cudagraph_max_bs * (num_speculative_tokens + 1)
    return _GDNSpecCausalConv1dHostBufferSlot(
        cache_indices_cpu=torch.empty(
            max_elements,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
        num_accepted_tokens_cpu=torch.empty(
            max_elements,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
    )


def _ensure_spec_causal_conv1d_host_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_spec_causal_conv1d_host_meta_initialized", False):
        return
    builder._ascend_gdn_spec_causal_conv1d_host_meta_initialized = True
    builder._ascend_gdn_spec_causal_conv1d_host_pool_idx = -1
    builder._ascend_gdn_spec_causal_conv1d_host_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_spec_causal_conv1d_host_pool = [
            _allocate_spec_causal_conv1d_host_slot(builder, device),
            _allocate_spec_causal_conv1d_host_slot(builder, device),
        ]


def _acquire_spec_causal_conv1d_host_slot(builder) -> _GDNSpecCausalConv1dHostBufferSlot:
    pool = builder._ascend_gdn_spec_causal_conv1d_host_pool
    builder._ascend_gdn_spec_causal_conv1d_host_pool_idx = (
        builder._ascend_gdn_spec_causal_conv1d_host_pool_idx + 1
    ) % len(pool)
    return pool[builder._ascend_gdn_spec_causal_conv1d_host_pool_idx]


def _copy_to_pinned_cpu(
    tensor: torch.Tensor,
    pinned_buffer: torch.Tensor | None,
) -> torch.Tensor:
    if tensor.device.type == "cpu":
        return tensor

    num_elements = tensor.numel()
    if pinned_buffer is None or pinned_buffer.numel() < num_elements:
        cpu_tensor = torch.empty(
            num_elements,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
    else:
        cpu_tensor = pinned_buffer[:num_elements]
    cpu_tensor.copy_(
        tensor.reshape(-1),
        non_blocking=True,
    )
    return cpu_tensor


def _build_non_spec_causal_conv1d_host_meta(
    builder,
    attn_metadata,
    non_spec_query_start_loc_cpu: torch.Tensor,
) -> GDNCausalConv1dHostMetadata:
    assert attn_metadata.num_prefills > 0
    if attn_metadata.non_spec_state_indices_tensor is None:
        raise RuntimeError("Expected attn_metadata.non_spec_state_indices_tensor for Ascend GDN non-spec prefill path.")
    if attn_metadata.has_initial_state is None:
        raise RuntimeError("Expected attn_metadata.has_initial_state for Ascend GDN non-spec prefill path.")

    slot = None
    if (
        attn_metadata.non_spec_state_indices_tensor.device.type != "cpu"
        or attn_metadata.has_initial_state.device.type != "cpu"
    ):
        slot = _acquire_causal_conv1d_host_slot(builder)

    cache_indices_cpu = _copy_to_pinned_cpu(
        attn_metadata.non_spec_state_indices_tensor,
        None if slot is None else slot.cache_indices_cpu,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    )
    has_initial_state_cpu = _copy_to_pinned_cpu(
        attn_metadata.has_initial_state,
        None if slot is None else slot.has_initial_state_cpu,
    )

    return GDNCausalConv1dHostMetadata(
        query_start_loc_cpu=non_spec_query_start_loc_cpu,
        cache_indices_cpu=cache_indices_cpu,
        has_initial_state_cpu=has_initial_state_cpu,
        _buffer_slot=slot,
    )


def _build_non_spec_decode_causal_conv1d_host_meta(
    builder,
    attn_metadata,
    non_spec_query_start_loc_cpu: torch.Tensor,
) -> GDNCausalConv1dHostMetadata:
    if attn_metadata.non_spec_state_indices_tensor is None:
        raise RuntimeError("Expected attn_metadata.non_spec_state_indices_tensor for Ascend GDN non-spec decode path.")

    slot = None
    if attn_metadata.non_spec_state_indices_tensor.device.type != "cpu":
        slot = _acquire_causal_conv1d_host_slot(builder)

    non_spec_cache_indices_cpu = _copy_to_pinned_cpu(
        attn_metadata.non_spec_state_indices_tensor,
        None if slot is None else slot.cache_indices_cpu,
    )

    return GDNCausalConv1dHostMetadata(
        query_start_loc_cpu=non_spec_query_start_loc_cpu,
        cache_indices_cpu=non_spec_cache_indices_cpu,
        has_initial_state_cpu=None,
        _buffer_slot=slot,
    )


def _build_spec_causal_conv1d_host_meta(
    builder,
    attn_metadata,
    spec_query_start_loc_cpu: torch.Tensor,
) -> GDNSpecCausalConv1dHostMetadata:
    assert attn_metadata.spec_sequence_masks is not None
    if attn_metadata.spec_state_indices_tensor is None:
        raise RuntimeError("Expected attn_metadata.spec_state_indices_tensor for Ascend GDN speculative path.")

    slot = None
    if attn_metadata.spec_state_indices_tensor.device.type != "cpu":
        slot = _acquire_spec_causal_conv1d_host_slot(builder)

    num_spec_decodes = attn_metadata.num_spec_decodes
    cache_indices_cpu = _copy_to_pinned_cpu(
        attn_metadata.spec_state_indices_tensor[:num_spec_decodes, 0].contiguous(),
        None if slot is None else slot.cache_indices_cpu,
    )

    num_accepted_tokens_cpu = _copy_to_pinned_cpu(
        attn_metadata.num_accepted_tokens[:num_spec_decodes].contiguous(),
        None if slot is None else slot.num_accepted_tokens_cpu,
    )

    return GDNSpecCausalConv1dHostMetadata(
        query_start_loc_cpu=spec_query_start_loc_cpu,
        cache_indices_cpu=cache_indices_cpu,
        num_accepted_tokens_cpu=num_accepted_tokens_cpu,
        _buffer_slot=slot,
    )


def _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
    shape_info = _build_chunk_meta_shape_info(builder, cu_seqlens_cpu)
    tensors = _allocate_chunk_meta_cpu_tensors(shape_info)
    _fill_chunk_meta_cpu_tensors(tensors, shape_info)
    return _build_chunked_prefill_metadata(builder, tensors, cu_seqlens_cpu=cu_seqlens_cpu)


def _build_non_spec_chunked_prefill_meta(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> GDNChunkedPrefillMetadata:
    device = builder._ascend_gdn_chunk_meta_device
    if device.type == "cpu":
        return _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu)

    shape_info = _build_chunk_meta_size_info(builder, cu_seqlens_cpu)
    builder._ascend_gdn_chunked_prefill_pool_idx = (builder._ascend_gdn_chunked_prefill_pool_idx + 1) % len(
        builder._ascend_gdn_chunked_prefill_pool
    )
    slot = builder._ascend_gdn_chunked_prefill_pool[builder._ascend_gdn_chunked_prefill_pool_idx]
    tensors = _slice_chunk_meta_slot_tensors(slot, shape_info)
    _fill_chunk_meta_device_tensors(builder, cu_seqlens, tensors)
    return _build_chunked_prefill_metadata(builder, tensors, cu_seqlens_cpu=cu_seqlens_cpu, slot=slot)


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    build_device_chunk_metadata = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sequence_index_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )

        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (sequence_index_capacity,), dtype=torch.bool, device=device
        )

        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )

        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
<<<<<<< HEAD
        self.spec_actual_seq_lengths: torch.Tensor | None = None
        self.non_spec_actual_seq_lengths: torch.Tensor | None = None
        if self._USE_COMMON_KERNEL_METADATA:
            self.spec_actual_seq_lengths = torch.empty((sequence_index_capacity + 1,), dtype=torch.int32, device=device)
            self.non_spec_actual_seq_lengths = torch.empty(
                (sequence_index_capacity + 1,), dtype=torch.int32, device=device
            )
=======
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold,
            supports_spec_as_decode,
            True,
        )
        if self.reorder_batch_threshold != 1:  # type: ignore
            speculative_config = self.vllm_config.speculative_config
            method = getattr(speculative_config, "method", None)
            num_spec = getattr(speculative_config, "num_speculative_tokens", None)
            if num_spec is not None and method in ("dflash", "dspark"):
                # The target-model verification query always contains the
                # base token plus N speculative tokens. DSpark's
                # sample_from_anchor only makes the draft-model forward use N
                # queries; it must not change target batch reordering.
                self.reorder_batch_threshold = 1 + num_spec

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

<<<<<<< HEAD
    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        if self.spec_actual_seq_lengths is not None:
            self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()

    def _attach_non_spec_prefill_metadata(
=======
    def _attach_non_spec_prefill_fallback_meta(
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
<<<<<<< HEAD
        attn_metadata.non_spec_prefill_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA or attn_metadata.num_prefills <= 0:
=======
        attn_metadata.non_spec_prefill_fallback_meta = None
        if attn_metadata.num_prefills <= 0:
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
            return attn_metadata

        _ensure_chunk_meta_state(self, common_attn_metadata.query_start_loc.device)
        _ensure_causal_conv1d_host_meta_state(
            self,
            common_attn_metadata.query_start_loc.device,
        )
        if non_spec_query_start_loc_cpu is None:
            raise RuntimeError("Expected non_spec_query_start_loc_cpu for Ascend GDN non-spec prefill path.")
        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")

        attn_metadata.non_spec_prefill_fallback_meta = GDNPrefillFallbackMeta(
            causal_conv1d=_build_non_spec_causal_conv1d_host_meta(
                self,
                attn_metadata,
                non_spec_query_start_loc_cpu,
            ),
            chunk=_build_non_spec_chunked_prefill_meta(
                self,
                non_spec_query_start_loc_cpu,
                attn_metadata.non_spec_query_start_loc,
            ),
        )
        return attn_metadata

    def _attach_spec_decode_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.spec_decode_fallback_meta = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        _ensure_spec_causal_conv1d_host_meta_state(
            self,
            common_attn_metadata.query_start_loc.device,
        )
        spec_query_start_loc_cpu = _build_spec_query_start_loc_cpu(
            self,
            common_attn_metadata,
            num_decode_draft_tokens_cpu,
        )
        if spec_query_start_loc_cpu is None:
            raise RuntimeError("Expected spec query_start_loc_cpu for Ascend GDN speculative path.")
        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.spec_query_start_loc for Ascend GDN speculative path.")

<<<<<<< HEAD
        num_sequences = attn_metadata.num_spec_decodes
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            num_sequences = attn_metadata.spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.spec_actual_seq_lengths
        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1

        attn_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=attn_metadata.spec_query_start_loc,
                cache_indices=attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
            ),
            actual_seq_lengths=(
                _build_actual_seq_lengths(
                    attn_metadata.spec_query_start_loc,
                    num_sequences,
                    actual_seq_lengths_buffer,
                )
                if self._USE_COMMON_KERNEL_METADATA
                else None
=======
        attn_metadata.spec_decode_fallback_meta = GDNSpecDecodeFallbackMeta(
            spec_causal_conv1d=_build_spec_causal_conv1d_host_meta(
                self,
                attn_metadata,
                spec_query_start_loc_cpu,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
            ),
        )
        return attn_metadata

    def _attach_non_spec_decode_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
<<<<<<< HEAD
        attn_metadata.non_spec_decode_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA:
            return attn_metadata
=======
        attn_metadata.non_spec_decode_fallback_meta = None
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        if attn_metadata.num_decodes <= 0:
            return attn_metadata

        _ensure_causal_conv1d_host_meta_state(
            self,
            common_attn_metadata.query_start_loc.device,
        )
        non_spec_query_start_loc_cpu = _build_non_spec_query_start_loc_cpu(
            self,
            attn_metadata,
            common_attn_metadata,
            num_decode_draft_tokens_cpu,
        )
        if non_spec_query_start_loc_cpu is None:
            raise RuntimeError("Expected non-spec query_start_loc_cpu for Ascend GDN non-spec decode path.")

        attn_metadata.non_spec_decode_fallback_meta = GDNDecodeFallbackMeta(
            causal_conv1d=_build_non_spec_decode_causal_conv1d_host_meta(
                self,
                attn_metadata,
                non_spec_query_start_loc_cpu,
            ),
        )
        return attn_metadata

<<<<<<< HEAD
    _SPEC_GRAPH_PAD_SLOT_ID = NULL_BLOCK_ID
    # Common GDN/KDA kernels consume prebuilt chunks and recurrent lengths.
    # 310P consumes top-level fields and only shares the spec conv wrapper.
    _USE_COMMON_KERNEL_METADATA = True

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize speculative decode metadata at graph request granularity."""
        actual_request_count = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        # Request and token buffers have different capacities under MTP.
        # Check token indices separately before modifying any captured inputs.
        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        if attn_metadata.spec_token_indx.numel() > self.spec_token_indx.numel():
            raise ValueError("Spec token indices exceed the graph token buffer capacity")
        if attn_metadata.non_spec_token_indx.numel() > self.non_spec_token_indx.numel():
            raise ValueError("Non-spec token indices exceed the graph token buffer capacity")

        attn_metadata.spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.spec_state_indices_tensor,
            spec_state_indices,
            actual_request_count,
            graph_request_count,
            self._SPEC_GRAPH_PAD_SLOT_ID,
        )
        attn_metadata.spec_sequence_masks = _materialize_graph_request_tensor(
            self.spec_sequence_masks,
            spec_sequence_masks,
            actual_request_count,
            graph_request_count,
            False,
        )

        non_spec_token_count = attn_metadata.non_spec_token_indx.size(0)
        spec_token_count = attn_metadata.spec_token_indx.size(0)
        self.non_spec_token_indx[:non_spec_token_count].copy_(
            attn_metadata.non_spec_token_indx,
            non_blocking=True,
        )
        self.spec_token_indx[:spec_token_count].copy_(
            attn_metadata.spec_token_indx,
            non_blocking=True,
        )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[:non_spec_token_count]
        attn_metadata.spec_token_indx = self.spec_token_indx[:spec_token_count]

        attn_metadata.spec_query_start_loc = _materialize_graph_query_start_loc(
            self.spec_query_start_loc,
            spec_query_start_loc,
            actual_request_count,
            graph_request_count,
        )
        attn_metadata.num_accepted_tokens = _materialize_graph_request_tensor(
            self.num_accepted_tokens,
            num_accepted_tokens,
            actual_request_count,
            graph_request_count,
            _GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT,
        )

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize non-speculative decode metadata at request granularity."""
        # Ordinary decode has one token per live request. Zero-length graph
        # rows (including idle DP batches) must not copy stale state indices.
        actual_request_count = attn_metadata.num_decode_tokens
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        attn_metadata.non_spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.non_spec_state_indices_tensor,
            state_indices,
            actual_request_count,
            graph_request_count,
            NULL_BLOCK_ID,
        )
        attn_metadata.non_spec_query_start_loc = _materialize_graph_query_start_loc(
            self.non_spec_query_start_loc,
            query_start_loc,
            actual_request_count,
            graph_request_count,
        )

=======
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> GDNAttentionMetadata:
<<<<<<< HEAD
        # Normalize FIA's full-width dummy queries before slicing requests:
        # its common token count may still include those inactive rows.
        if self.use_full_cuda_graph and self.use_spec_decode:
            common_attn_metadata = _remove_spec_graph_padding_queries(
                common_attn_metadata,
                num_decode_draft_tokens_cpu,
            )
        m = common_attn_metadata
        # Common metadata follows the padded graph shape. Build request-phase
        # metadata from the logical batch, then materialize graph-sized buffers.
        graph_request_count = m.num_reqs
        if num_actual_reqs is None:
            num_actual_reqs = graph_request_count
        elif not 0 <= num_actual_reqs <= graph_request_count:
            raise ValueError(
                f"num_actual_reqs ({num_actual_reqs}) must be between 0 and graph_request_count ({graph_request_count})"
            )

        if num_actual_reqs < graph_request_count:
            m = m.unpadded(m.num_actual_tokens, num_actual_reqs)
            if num_accepted_tokens is not None:
                num_accepted_tokens = num_accepted_tokens[:num_actual_reqs]
            if num_decode_draft_tokens_cpu is not None:
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu[:num_actual_reqs]
        m = _treat_single_token_prefills_with_state_as_decodes(m)
=======
        m = common_attn_metadata
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
<<<<<<< HEAD
            runtime_draft_tokens = num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            if runtime_draft_tokens.sum().item() > 0:
                torch.ge(
                    num_decode_draft_tokens_cpu,
                    0,
                    out=spec_sequence_masks_cpu,
                )
            else:
                # Dynamic speculative decoding can be enabled while this batch
                # carries no draft tokens. Keep the normal prefill/decode split.
                spec_sequence_masks_cpu.zero_()
=======
            torch.ge(
                num_decode_draft_tokens_cpu,
                0,
                out=spec_sequence_masks_cpu,
            )
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                m,
                decode_threshold=1,
                treat_short_extends_as_decodes=False,
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_indices is not None
            assert non_spec_sequence_indices is not None

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens

            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, : self.num_spec + 1],
                    0,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, : self.num_spec + 1],
                    0,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, 0],
                    0,
                    non_spec_sequence_indices,
                )
                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = torch.index_select(
                num_accepted_tokens,
                0,
                spec_sequence_indices,
            )

        # A FULL graph retains captured speculative conv/recurrent tasks. Clear
        # their stable inputs on every no-spec replay so an idle or prefill
        # batch cannot mutate state belonging to the preceding request.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            self._reset_spec_decode_graph_inputs(graph_request_count)

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
<<<<<<< HEAD
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            assert prefill_query_start_loc_cpu is not None
            if self._USE_COMMON_KERNEL_METADATA:
                non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                    self,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                    build_device_metadata=self.build_device_chunk_metadata,
                )
                # Common GDN/KDA callers also consume the upstream chunk fields.
                chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
                chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

        if num_prefills > 0:
            has_initial_state = self._build_prefill_has_initial_state(
                context_lens_tensor=m.compute_num_computed_tokens(),
                non_spec_sequence_indices=non_spec_sequence_indices,
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
=======
            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            gpu_device = query_start_loc.device
            chunk_indices = prepare_chunk_indices(
                non_spec_query_start_loc_cpu,
                FLA_CHUNK_SIZE,
            ).to(device=gpu_device, non_blocking=True)
            chunk_offsets = prepare_chunk_offsets(
                non_spec_query_start_loc_cpu,
                FLA_CHUNK_SIZE,
            ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state, nums_dict, batch_ptr, token_chunk_offset_ptr = (
                self._build_prefill_has_initial_state_and_causal_conv1d_meta(
                    common_attn_metadata=m,
                    context_lens_tensor=context_lens_tensor,
                    num_prefills=num_prefills,
                    spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                    non_spec_sequence_indices=non_spec_sequence_indices,
                    non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
                    query_start_loc=query_start_loc,
                )
            )
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        else:
            has_initial_state = None

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

<<<<<<< HEAD
=======
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[batch_size:].fill_(NULL_BLOCK_ID)
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor,
                non_blocking=True,
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes],
                non_blocking=True,
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx,
                non_blocking=True,
            )
            non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx,
                non_blocking=True,
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc,
                non_blocking=True,
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens,
                non_blocking=True,
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[batch_size:].fill_(NULL_BLOCK_ID)
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor,
                non_blocking=True,
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc,
                non_blocking=True,
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
        )
<<<<<<< HEAD
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and self._can_pad_spec_decode(graph_request_count, num_spec_decode_tokens)
        ):
            self._pad_spec_decode_metadata(attn_metadata, graph_request_count)
        elif (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and graph_request_count <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_request_count)
            non_spec_conv1d_cache_indices = attn_metadata.non_spec_state_indices_tensor
        # Attach once, using the final graph-stable Tensor references.
        self._attach_non_spec_prefill_metadata(
            attn_metadata, non_spec_chunked_prefill_metadata, non_spec_conv1d_cache_indices
=======
        attn_metadata = self._attach_non_spec_prefill_fallback_meta(
            attn_metadata,
            common_attn_metadata,
            non_spec_query_start_loc_cpu,
        )
        attn_metadata = self._attach_spec_decode_fallback_meta(
            attn_metadata,
            common_attn_metadata,
            num_decode_draft_tokens_cpu,
        )
        return self._attach_non_spec_decode_fallback_meta(
            attn_metadata,
            common_attn_metadata,
            num_decode_draft_tokens_cpu,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        )
        self._attach_spec_decode_metadata(attn_metadata)
        self._attach_non_spec_decode_metadata(attn_metadata, non_spec_conv1d_cache_indices)
        return attn_metadata

    def _build_prefill_has_initial_state(
        self,
        *,
        context_lens_tensor: torch.Tensor,
        non_spec_sequence_indices: torch.Tensor | None,
<<<<<<< HEAD
    ) -> torch.Tensor:
=======
        non_spec_query_start_loc_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        dict[int, dict[str, object]] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del common_attn_metadata, num_prefills
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        has_initial_state = context_lens_tensor > 0
        if non_spec_sequence_indices is not None:
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
<<<<<<< HEAD
        return has_initial_state
=======
            assert non_spec_query_start_loc_cpu is not None
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            non_spec_query_start_loc_cpu,
            device=query_start_loc.device,
        )
        return has_initial_state, nums_dict, batch_ptr, token_chunk_offset_ptr
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder


class AscendGDNHostMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build only the descriptors consumed by AscendC host-metadata paths."""

    build_device_chunk_metadata = False


class AscendGDNHostMetadataBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNHostMetadataBuilder]:
        return AscendGDNHostMetadataBuilder


class AscendGDNFusedAttentionBackend(AscendGDNAttentionBackend):
    """Select host-only metadata when Qwen can use a fused prefill op."""

    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        # Import lazily to avoid the gdn <-> builder import cycle at module
        # initialization.  The probe result is cached by the attention class.
        from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

        if get_pcp_group().world_size == 1 and AscendGatedDeltaNetAttention._supports_host_metadata_prefill():
            return AscendGDNHostMetadataBuilder
        return AscendGDNAttentionMetadataBuilder
