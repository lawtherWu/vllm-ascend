#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_npu
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    CPKVScope,
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    use_history_current_split_decode,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
<<<<<<< HEAD
    enable_dcp,
=======
    filter_chunked_req_indices,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    split_decodes_and_prefills,
)
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine
from vllm_ascend.utils import (
    cp_chunkedprefill_comm_stream,
    cp_decode_comm_stream,
    is_pd_decode_recompute_scheduler_enabled,
)
<<<<<<< HEAD

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext
=======
from vllm_ascend.utils import cp_chunkedprefill_comm_stream, weak_ref_tensors
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


@dataclass
class AscendMetadataForPrefill:
    """GQA prefill metadata used only by DCP."""

    @dataclass
    class ChunkedContextMetadata:
        actual_chunk_seq_lengths: list[int]
        actual_seq_lengths_kv: list[int]
        starts: torch.Tensor
        local_context_lens: torch.Tensor | None = None
        local_total_toks: int | None = None
        pcp_query_restore_idx: torch.Tensor | None = None
        pcp_local_query_indices: torch.Tensor | None = None

    chunked_context: ChunkedContextMetadata | None = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: list[int] | None = None
    pcp_current_kv_indices: torch.Tensor | None = None
    pcp_actual_seq_lengths_kv: list[int] | None = None


@dataclass
class AscendMetadataForDecode:
    """GQA decode metadata used only by DCP."""

    num_computed_tokens_of_dcp: np.ndarray | None = None
    block_tables: torch.Tensor = None
    cp_history_seq_len: list[int] | None = None
    actual_seq_lengths_q: list[int] | None = None
    seq_lens_list: list[int] | None = None

    def update_dcp_seq_lens_cpu(
        self,
        seq_lens_cpu: torch.Tensor,
        dcp_local_seq_lens_cpu: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        *,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int,
    ) -> None:
        """Partition history after removing the global current-token chunk."""
        num_computed_tokens_of_dcp: np.ndarray = get_dcp_local_seq_lens(
            seq_lens_cpu,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ).numpy()
        num_computed_tokens_of_dcp[:, dcp_rank] = 0
        num_computed_tokens_of_dcp[: dcp_local_seq_lens_cpu.numel(), dcp_rank] = dcp_local_seq_lens_cpu.numpy()
        self.num_computed_tokens_of_dcp = num_computed_tokens_of_dcp
        self.cp_history_seq_len = get_dcp_local_seq_lens(
            (seq_lens_cpu - query_lens_cpu).clamp(min=0),
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ).tolist()


@dataclass
class AscendAttentionDCPMetadata(AscendMetadata):
    """GQA metadata fields used only by the DCP execution path."""

    prefill: AscendMetadataForPrefill | None = None
    decode: AscendMetadataForDecode | None = None


class AscendAttentionDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendAttentionMetadataBuilder,
):
    """Build attention metadata for decode context parallelism."""

    metadata_cls = AscendAttentionDCPMetadata
    consumes_pcp_context = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dcp_enabled = enable_dcp()
        self.pcp_group = get_pcp_group()
        self._pcp_context: AscendPCPAttentionContext | None = None
        self._pcp_cache_group_idx: int | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        *,
        pcp_context: "AscendPCPAttentionContext | None" = None,
        pcp_cache_group_idx: int | None = None,
    ) -> AscendAttentionDCPMetadata:
        self._pcp_context = pcp_context
        self._pcp_cache_group_idx = pcp_cache_group_idx
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert isinstance(metadata, AscendAttentionDCPMetadata)
        return metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendAttentionDCPMetadata:
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata, **kwargs)

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=(
                not self.pcp_enabled
                and (
                    self.speculative_config is not None
                    or (self.dcp_enabled and is_pd_decode_recompute_scheduler_enabled(self.vllm_config))
                )
            ),
        )

    def _build_pcp_prefill_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> AscendMetadataForPrefill:
        """Build current KV prefixes for local Q fragments and separate history metadata."""
        context = self._pcp_context
        if context is None or context.local_to_global_req_indices is None:
            raise RuntimeError("GQA PCP+DCP prefill requires the global request view and local request mapping.")
        batch = context.global_batch
        global_num_decodes = int((~batch.is_prefilling_np[: batch.num_reqs]).sum())
        global_decode_tokens = int(batch.query_start_loc_np[global_num_decodes])
        global_num_tokens = int(batch.query_start_loc_np[batch.num_reqs])
        local_tokens = common_attn_metadata.slot_mapping.numel() // self.pcp_group.world_size
        local_decode_tokens = int(query_lens[:num_decodes].sum())

        # Compress [decode, prefill] per rank into the prefill-only Q gather layout.
        restore_idx = context.hidden_restore_idx[global_decode_tokens:global_num_tokens]
        query_restore_idx = (
            (restore_idx // local_tokens) * (local_tokens - local_decode_tokens)
            + restore_idx % local_tokens
            - local_decode_tokens
        ).to(torch.int64)
        # Gathered current KV retains one shared decode area before prefill KV.
        current_kv_indices = query_restore_idx + local_decode_tokens

        rows = context.local_to_global_req_indices[num_decodes : num_decodes + num_prefills]
        if len(rows) != num_prefills:
            raise RuntimeError("GQA prefill request mapping does not match the local fragment count.")
        if common_attn_metadata.causal:
            history_lens = torch.as_tensor(batch.num_computed_tokens_np[list(rows)], dtype=seq_lens.dtype)
            current_lens = seq_lens[num_decodes:] - history_lens
        else:
            # Noncausal Q fragments must see the entire current chunk of their request.
            global_query_lens = np.diff(batch.query_start_loc_np[: batch.num_reqs + 1])
            current_lens = torch.as_tensor(global_query_lens[list(rows)], dtype=seq_lens.dtype)
        starts = batch.query_start_loc_np[list(rows)] - global_decode_tokens
        prefixes = [current_kv_indices[start : start + length] for start, length in zip(starts, current_lens.tolist())]

<<<<<<< HEAD
        chunked_context = self._build_pcp_chunked_context(
            common_attn_metadata, global_num_decodes, local_decode_tokens, query_restore_idx
        )
        if chunked_context is not None:
            assert self._pcp_cache_group_idx is not None
            block_table = context.global_block_tables[self._pcp_cache_group_idx][global_num_decodes : batch.num_reqs]
=======
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded]
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)
        attn_state = common_attn_metadata.attn_state
        num_computed_tokens_cpu = seq_lens - query_lens

        query_start_loc = query_start_loc_cpu.to(self.device, non_blocking=True)

        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        prefill_metadata = None
        decode_metadata = None
        if common_long_seq_metadata is None:
            raise AssertionError("common_long_seq_metadata should not be None.")
        num_computed_tokens_of_pcp_dcp = common_long_seq_metadata.num_computed_tokens_of_pcp_dcp
        assert num_computed_tokens_of_pcp_dcp is not None
        chunked_context_metadata = None
        attn_mask_seqlens = common_long_seq_metadata.attn_mask_seqlens
        attn_chunk_seqlens = common_long_seq_metadata.attn_chunk_seqlens
        if num_prefills > 0:
            query_lens = query_lens[num_decodes:]
            context_lens_cpu = num_computed_tokens_cpu[num_decodes:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                if self.pcp_size > 1 and common_long_seq_metadata.pcp_use_hybrid_attn:
                    query_lens = attn_chunk_seqlens * 2
                local_context_lens_allranks = (
                    torch.tensor(num_computed_tokens_of_pcp_dcp)[num_decodes:].to(self.device).to(dtype=torch.int32)
                )
                local_chunked_kv_lens_rank = local_context_lens_allranks[:, self.pcp_rank, self.dcp_rank]
                actual_seq_lengths_kv = torch.cumsum(local_chunked_kv_lens_rank, dim=0).tolist()
                local_total_toks = local_chunked_kv_lens_rank.sum()
                chunked_req_mask = self._get_chunked_req_mask(local_context_lens_allranks)
                local_chunk_starts = torch.zeros(
                    (len(local_context_lens_allranks),), dtype=torch.int32, device=self.device
                )
                # Note(qcs): we only do restore and recover for pcp, and set these vars to None
                # when only using dcp.
                if self.pcp_size > 1:
                    kv_inverse_idx_for_chunk = torch.argsort(
                        common_long_seq_metadata.pcp_allgather_restore_idx[self.pcp_size * num_decode_tokens :].to(
                            torch.float32
                        )
                    )
                    cp_kv_recover_idx_for_chunk = torch.argsort(kv_inverse_idx_for_chunk)
                else:
                    kv_inverse_idx_for_chunk = None
                    cp_kv_recover_idx_for_chunk = None

                chunk_seq_mask_filtered_indices = filter_chunked_req_indices(query_lens, chunked_req_mask).to(
                    self.device
                )
                chunked_context_metadata = AscendMetadataForPrefill.ChunkedContextMetadata(
                    actual_chunk_seq_lengths=torch.cumsum(query_lens * self.pcp_size, dim=0),
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    chunked_req_mask=chunked_req_mask,
                    starts=local_chunk_starts,
                    local_context_lens_allranks=local_context_lens_allranks,
                    cp_kv_recover_idx_for_chunk=cp_kv_recover_idx_for_chunk,
                    kv_inverse_idx_for_chunk=kv_inverse_idx_for_chunk,
                    chunk_seq_mask_filtered_indices=chunk_seq_mask_filtered_indices,
                    local_total_toks=local_total_toks.item(),
                )
            head_attn_nomask_seqlens = common_long_seq_metadata.head_attn_nomask_seqlens
            tail_attn_nomask_seqlens = common_long_seq_metadata.tail_attn_nomask_seqlens

            pcp_metadata = AscendPCPMetadata(
                q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
                q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
                kv_with_q_head_nomask_idx=common_long_seq_metadata.kv_with_q_head_nomask_idx_tensor,
                kv_with_q_head_mask_idx=common_long_seq_metadata.kv_with_q_head_mask_idx_tensor,
                kv_with_q_tail_nomask_idx=common_long_seq_metadata.kv_with_q_tail_nomask_idx_tensor,
                kv_with_q_tail_mask_idx=common_long_seq_metadata.kv_with_q_tail_mask_idx_tensor,
                attn_mask_seqlens=attn_mask_seqlens,
                head_attn_nomask_seqlens=head_attn_nomask_seqlens,
                tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
                q_full_idx=common_long_seq_metadata.q_full_idx,
                pcp_use_hybrid_attn=common_long_seq_metadata.pcp_use_hybrid_attn,
                pcp_unpad_mask=common_long_seq_metadata.pcp_unpad_mask,
                pcp_allgather_restore_idx=common_long_seq_metadata.pcp_allgather_restore_idx,
                pcp_fa_query_idx=common_long_seq_metadata.pcp_fa_query_idx,
                pcp_padded_tokens_fla=common_long_seq_metadata.pcp_padded_tokens_fla,
                pcp_enter_fa_restore_idx=common_long_seq_metadata.pcp_enter_fa_restore_idx,
                max_num_tokens_across_pcp=common_long_seq_metadata.max_num_tokens_across_pcp,
                total_num_scheduled_tokens=common_long_seq_metadata.total_num_scheduled_tokens,
            )

            prefill_metadata = AscendMetadataForPrefill(
                pcp_metadata=pcp_metadata,
                pcp_exit_fa_scatter_idx=common_long_seq_metadata.pcp_exit_fa_scatter_idx,
                chunked_context=chunked_context_metadata,
                block_tables=block_table[num_decodes:, ...],
                actual_seq_lengths_q=torch.cumsum(query_lens, dim=0),
            )

        if num_decodes > 0:
            num_computed_tokens_array = np.array(num_computed_tokens_of_pcp_dcp)
            num_computed_tokens_array = num_computed_tokens_array[:num_decodes]
            # TODO: numpy array mode of the shared memory is used to improve performance
            if common_long_seq_metadata.dcp_mtp_attn_mask is not None:
                dcp_mtp_attn_mask = common_long_seq_metadata.dcp_mtp_attn_mask
            else:
                dcp_mtp_attn_mask = None
            decode_metadata = AscendMetadataForDecode(
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_array,
                block_tables=block_table[:num_decodes],
                dcp_mtp_attn_mask=dcp_mtp_attn_mask,
            )

        if self.decode_threshold == 1:
            actual_seq_lengths_q = (torch.arange(num_decodes) + 1).tolist() + query_start_loc_cpu[1:].tolist()[
                num_decodes:
            ]
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        else:
            block_table = block_table[num_decodes:]
        return AscendMetadataForPrefill(
            block_tables=block_table,
            actual_seq_lengths_q=query_lens[num_decodes:].cumsum(0).tolist(),
            pcp_current_kv_indices=torch.cat(prefixes),
            pcp_actual_seq_lengths_kv=current_lens.cumsum(0).tolist(),
            chunked_context=chunked_context,
        )

    def _build_chunked_context_metadata(
        self,
        query_lens: torch.Tensor,
        context_lens_cpu: torch.Tensor,
        *,
        context_lens: torch.Tensor | None = None,
    ) -> AscendMetadataForPrefill.ChunkedContextMetadata:
        """Build shared history fields while keeping host FIA lengths and device cache-load lengths separate."""
        interleave = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        local_context_lens_cpu = get_dcp_local_seq_lens(
            context_lens_cpu, dcp_size=self.dcp_size, dcp_rank=self.dcp_rank, cp_kv_cache_interleave_size=interleave
        )
        if context_lens is None:
            local_context_lens = local_context_lens_cpu.to(self.device, non_blocking=True)
        else:
            local_context_lens = get_dcp_local_seq_lens(
                context_lens, dcp_size=self.dcp_size, dcp_rank=self.dcp_rank, cp_kv_cache_interleave_size=interleave
            )
        return AscendMetadataForPrefill.ChunkedContextMetadata(
            actual_chunk_seq_lengths=query_lens.cumsum(0).tolist(),
            actual_seq_lengths_kv=local_context_lens_cpu.cumsum(0).tolist(),
            starts=torch.zeros(query_lens.numel(), dtype=torch.int32, device=self.device),
            local_context_lens=local_context_lens,
            local_total_toks=int(local_context_lens_cpu.sum()),
        )

    def _build_pcp_chunked_context(
        self,
<<<<<<< HEAD
        common_attn_metadata: AscendCommonAttentionMetadata,
        global_num_decodes: int,
        local_decode_tokens: int,
        query_restore_idx: torch.Tensor,
    ) -> AscendMetadataForPrefill.ChunkedContextMetadata | None:
        """Prepare the global request view and PCP query indices for historical attention."""
        context = self._pcp_context
        assert context is not None
        batch = context.global_batch
        history_lens = torch.from_numpy(batch.num_computed_tokens_np[global_num_decodes : batch.num_reqs].copy()).int()
        if not history_lens.any():
            return None
        if self._pcp_cache_group_idx is None or context.padded_gather_idx is None:
            raise RuntimeError("GQA history attention requires PCP's global cache table and query layout.")
        query_lens = torch.from_numpy(
            np.diff(batch.query_start_loc_np[global_num_decodes : batch.num_reqs + 1]).copy()
        ).int()
        metadata = self._build_chunked_context_metadata(query_lens, history_lens)
        local_tokens = common_attn_metadata.slot_mapping.numel() // self.pcp_group.world_size
        rank_start = self.pcp_group.rank_in_group * local_tokens
        local_query_slice = slice(rank_start + local_decode_tokens, rank_start + common_attn_metadata.num_actual_tokens)
        global_decode_tokens = int(batch.query_start_loc_np[global_num_decodes])
        metadata.pcp_query_restore_idx = query_restore_idx
        metadata.pcp_local_query_indices = (context.padded_gather_idx[local_query_slice] - global_decode_tokens).to(
            torch.int64
        )
        return metadata

    def _build_backend_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, object]:
        prefill_metadata = None
        if num_prefills > 0:
            if self.pcp_enabled:
                prefill_metadata = self._build_pcp_prefill_metadata(
                    common_attn_metadata, block_table, query_lens, seq_lens, num_decodes, num_prefills
                )
            else:
                prefill_query_lens = query_lens[num_decodes:]
                prefill_query_ends = torch.cumsum(prefill_query_lens, dim=0)
                context_lens_cpu = (seq_lens - query_lens)[num_decodes:]
                chunked_context_metadata = None
                if self.chunked_prefill_enabled and context_lens_cpu.numel() > 0 and context_lens_cpu.max().item() > 0:
                    prefill_end = num_decodes + num_prefills
                    query_start_loc = common_attn_metadata.query_start_loc[num_decodes : prefill_end + 1]
                    context_lens = common_attn_metadata.seq_lens[num_decodes:prefill_end] - torch.diff(query_start_loc)
                    chunked_context_metadata = self._build_chunked_context_metadata(
                        prefill_query_lens, context_lens_cpu, context_lens=context_lens
                    )
                prefill_metadata = AscendMetadataForPrefill(
                    chunked_context=chunked_context_metadata,
                    block_tables=block_table[num_decodes:],
                    actual_seq_lengths_q=prefill_query_ends.tolist(),
                )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AscendMetadataForDecode(
                block_tables=block_table[:num_decodes],
                actual_seq_lengths_q=query_lens[:num_decodes].cumsum(0).tolist(),
                seq_lens_list=seq_lens[:num_decodes].tolist(),
            )
            dcp_local_seq_lens_cpu = common_attn_metadata.dcp_local_seq_lens_cpu
            assert dcp_local_seq_lens_cpu is not None
            decode_metadata.update_dcp_seq_lens_cpu(
                seq_lens[:num_decodes],
                dcp_local_seq_lens_cpu[:num_decodes],
                query_lens[:num_decodes],
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )

        return {
            "prefill": prefill_metadata,
            "decode": decode_metadata,
        }
=======
        q: torch.Tensor,
        q_seqlens: list[int],
        k_nomask: torch.Tensor,
        v_nomask: torch.Tensor,
        kv_seqlens_nomask: list[int],
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
        kv_seqlens_mask: list[int],
        mask: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # nomask Attention
        if k_nomask is not None:
            attn_out_nomask, attn_lse_nomask = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                k_nomask,
                v_nomask,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=self.scale,
                sparse_mode=0,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=kv_seqlens_nomask,
                actual_seq_lengths=q_seqlens,
            )

        # mask Attention
        attn_out_mask, attn_lse_mask = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k_mask,
            v_mask,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=mask,
            scale=self.scale,
            sparse_mode=3,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens,
        )
        # update
        output = attn_out_mask
        attn_lse = attn_lse_mask
        if k_nomask is not None:
            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is None:
                output = _npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask)
                attn_lse = None
            else:
                output, attn_lse = _update_out_and_lse(
                    torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                    torch.stack([attn_lse_nomask, attn_lse_mask], dim=0),
                )
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)


@dataclass(frozen=True, slots=True)
class DCPFIAParamProvider:
    metadata_layer_name: str | None
    dcp_rank: int
    attention_kind: CPKVScope

    @property
    def layer_name(self) -> tuple[str | None, CPKVScope]:
        # Draft SharedSource matches each captured task independently.
        return self.metadata_layer_name, self.attention_kind

    def resolve(self, attn_metadata) -> dict[str, object]:
        metadata = attn_metadata[self.metadata_layer_name]
        decode = metadata.decode
        assert decode is not None
        query_lens = decode.actual_seq_lengths_q
        assert query_lens is not None
        if self.attention_kind == CPKVScope.CURRENT:
            kv_lens = query_lens
            block_table = None
        elif self.attention_kind == CPKVScope.HISTORY:
            assert decode.cp_history_seq_len is not None
            kv_lens = decode.cp_history_seq_len
            block_table = decode.block_tables
        else:
            kv_lens = decode.num_computed_tokens_of_dcp[:, self.dcp_rank].tolist()
            block_table = decode.block_tables
        return {
            "actual_seq_lengths": query_lens,
            "actual_seq_lengths_kv": kv_lens,
            "block_table": block_table,
        }


def build_dcp_fia_params(
    layer_name: str,
    metadata,
    dcp_rank: int,
    *,
    is_draft_model: bool = False,
    is_draft_model_prefill: bool = False,
    use_spec_decode: bool = False,
) -> list[dict[str, object]]:
    """Publish parameters for the captured split or ordinary cache task."""
    use_split = use_history_current_split_decode(
        metadata,
        is_draft_model=is_draft_model,
        is_draft_model_prefill=is_draft_model_prefill,
        use_spec_decode=use_spec_decode,
    )
    kinds = (CPKVScope.HISTORY, CPKVScope.CURRENT) if use_split else (CPKVScope.FULL,)
    params = []
    for kind in kinds:
        provider = DCPFIAParamProvider(layer_name, dcp_rank, kind)
        params.append({"layer_name": provider.layer_name, **provider.resolve({layer_name: metadata})})
    return params


class AscendAttentionDCPImpl(DCPImplMixin, AscendAttentionBackendImpl):
    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcp_enabled = self.vllm_config.parallel_config.prefill_context_parallel_size > 1

    def _run_dcp_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
        attention_kind: CPKVScope,
        num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = DCPFIAParamProvider(self._graph_metadata_layer_name(), self.dcp_rank, attention_kind)
        params = provider.resolve({provider.metadata_layer_name: attn_metadata})
        is_current = attention_kind == CPKVScope.CURRENT
        kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": "TND",
            "atten_mask": attn_metadata.attn_mask if is_current and attn_metadata.causal else None,
            "sparse_mode": 3 if is_current and attn_metadata.causal else 0,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            **params,
        }
        if not is_current:
            assert self.key_cache is not None
            kwargs["block_size"] = self.key_cache.shape[1]
            kwargs["inner_precise"] = 1
        if not _EXTRA_CTX.capturing:
            return torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)

        # Match attention_v1: keep addresses stable and update only runtime
        # lengths/block tables through UpdatableGraph parameter providers.
        workspace = get_capture_resource(
            (DCPFIAParamProvider, attention_kind, num_heads, self.num_kv_heads),
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(query, key, value, **kwargs),
            self._use_max_workspace_for_fia_graph,
        )
        output = torch.empty_like(query)
        lse = torch.empty((query.shape[0], num_heads, 1), dtype=torch.float32, device=query.device)
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {
                "query": query,
                "key": key,
                "value": value,
                **kwargs,
                "workspace": workspace,
                "out": [output, lse],
            },
            provider,
        )
        return output, lse

    def _forward_decode_dcp(
        self,
        query: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
        current_key: torch.Tensor | None = None,
        current_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        (history_query,) = self._dcp_all_gather_fragments(query, dim=1)
        num_heads = history_query.shape[1]
        key = self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.value_cache.shape[0], self.value_cache.shape[1], -1)
        use_split = use_history_current_split_decode(
            attn_metadata,
            is_draft_model=_EXTRA_CTX.is_draft_model,
            is_draft_model_prefill=_EXTRA_CTX.is_draft_model_prefill,
            use_spec_decode=self.vllm_config.speculative_config is not None,
        )
        kind = CPKVScope.HISTORY if use_split else CPKVScope.FULL
        history_output, history_lse = self._run_dcp_attention(history_query, key, value, attn_metadata, kind, num_heads)
        if not use_split:
            return self._merge_dcp_attention_output(history_output, history_lse)

        assert current_key is not None and current_value is not None
        main_stream = torch.npu.current_stream()
        attn_stream = cp_decode_comm_stream()
        history_ready = main_stream.record_event()
        for tensor in (query, current_key, current_value, attn_metadata.attn_mask):
            if tensor is not None:
                tensor.record_stream(attn_stream)
        # Match MLA: current attention overlaps history packing and A2A.
        with torch.npu.stream(attn_stream):
            attn_stream.wait_event(history_ready)
            current_output, current_lse = self._run_dcp_attention(
                query,
                current_key.contiguous(),
                current_value.contiguous(),
                attn_metadata,
                CPKVScope.CURRENT,
                self.num_heads,
            )
            current_attn_done = attn_stream.record_event()
        current_output.record_stream(main_stream)
        current_lse.record_stream(main_stream)
        history_recv = self._merge_dcp_attention_output(
            history_output,
            history_lse,
            defer_combine=True,
        )
        main_stream.wait_event(current_attn_done)
        # Only historical shards participate in A2A. Current K/V are
        # replicated across DCP ranks and must contribute exactly once.
        return fused_dcp_lse_combine(
            history_recv,
            self.head_size,
            scatter_dim=1,
            local_output=current_output,
            local_lse=current_lse,
        )

    def _prefill_query_all_gather(self, attn_metadata, prefill_query):
        if self.pcp_enabled:
            prefill = attn_metadata.prefill
            assert prefill is not None
            chunked_context = prefill.chunked_context
            assert chunked_context is not None and chunked_context.pcp_query_restore_idx is not None
            # PCP splits the query sequence, not the history KV cache. Restore
            # all current queries before computing each DCP history shard.
            prefill_query = self.pcp_group.all_gather(prefill_query.contiguous(), dim=0)
            prefill_query = torch.index_select(prefill_query, 0, chunked_context.pcp_query_restore_idx)
        (prefill_query,) = self._dcp_all_gather_fragments(prefill_query, dim=1)
        return prefill_query

    def _compute_prefill_context(
        self,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendAttentionDCPMetadata,
    ):
        assert len(kv_cache) > 1
        assert attn_metadata is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        local_chunked_kv_lens_rank = prefill_metadata.chunked_context.local_context_lens
        assert local_chunked_kv_lens_rank is not None
        total_toks = prefill_metadata.chunked_context.local_total_toks
        key, value = self._load_kv_for_chunk(attn_metadata, kv_cache, local_chunked_kv_lens_rank, query, total_toks)
        num_heads = query.shape[1]

        if total_toks == 0:
            return (
                torch.full(
                    (query.size(0), num_heads, self.head_size), fill_value=0, dtype=query.dtype, device=query.device
                ),
                torch.full(
                    (query.size(0), num_heads, 1), fill_value=-torch.inf, dtype=torch.float32, device=query.device
                ),
            )

        prefix_chunk_output, prefix_chunk_lse = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=None,
            scale=self.scale,
            sparse_mode=0,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=prefill_metadata.chunked_context.actual_seq_lengths_kv,
            actual_seq_lengths=prefill_metadata.chunked_context.actual_chunk_seq_lengths,
        )

        return prefix_chunk_output, prefix_chunk_lse

    def _forward_prefill_current_kv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute local Q against current KV, selecting PCP causal prefixes when needed."""
        prefill = attn_metadata.prefill
        assert prefill is not None and prefill.actual_seq_lengths_q is not None
        query_lens = prefill.actual_seq_lengths_q
        local_query = query[attn_metadata.num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()
        causal = attn_metadata.causal
        if self.pcp_enabled:
            assert prefill.pcp_current_kv_indices is not None and prefill.pcp_actual_seq_lengths_kv is not None
            key = torch.index_select(key, 0, prefill.pcp_current_kv_indices)
            value = torch.index_select(value, 0, prefill.pcp_current_kv_indices)
            kv_lens = prefill.pcp_actual_seq_lengths_kv
        else:
            key = key[attn_metadata.num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()
            value = value[attn_metadata.num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()
            kv_lens = query_lens
        record_attention_compute_start()
        attn_output, attn_lse = torch_npu.npu_fused_infer_attention_score(
            local_query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=attn_metadata.attn_mask if causal else None,
            sparse_mode=3 if causal else 0,
            scale=self.scale,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths=query_lens,
            actual_seq_lengths_kv=kv_lens,
        )
        return attn_output.view_as(local_query), attn_lse

    def _load_kv_for_chunk(self, attn_metadata, kv_cache, local_chunked_kv_lens_rank, query, total_toks):
        cache_key = kv_cache[0]
        cache_value = kv_cache[1]
        num_heads = cache_key.size(2)
        head_size = kv_cache[0].size(-1)

        key = torch.empty(total_toks, num_heads, head_size, dtype=query.dtype, device=query.device)
        value = torch.empty(total_toks, num_heads, head_size, dtype=query.dtype, device=query.device)
        if total_toks > 0:
            DeviceOperator.kv_cache_load(
                cache_key,
                cache_value,
                attn_metadata.prefill.block_tables,
                local_chunked_kv_lens_rank,
                # slot offsets of current chunk in current iteration
                attn_metadata.prefill.chunked_context.starts,
                key=key,
                value=value,
            )
        return key, value

<<<<<<< HEAD
=======
    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        output_padded = output

        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if has_decode:
                slot_mapping = attn_metadata.slot_mapping[: num_decode_tokens * self.pcp_size : self.pcp_size]
                DeviceOperator.reshape_and_cache(
                    key=key[:num_decode_tokens],
                    value=value[:num_decode_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_mapping=slot_mapping,
                )

            if has_prefill:
                if self.pcp_size > 1:
                    assert attn_metadata.prefill is not None and attn_metadata.prefill.pcp_metadata is not None
                    if not attn_metadata.prefill.pcp_metadata.pcp_use_hybrid_attn:
                        kv = torch.cat([key, value], dim=-1)
                        num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
                        all_kv = get_pcp_group().all_gather(kv[:num_actual_tokens_pcp_padded].contiguous(), dim=0)
                        pcp_allgather_restore_idx = attn_metadata.prefill.pcp_metadata.pcp_allgather_restore_idx
                        all_kv = torch.index_select(all_kv, 0, pcp_allgather_restore_idx)
                        key, value = all_kv.split([self.head_size, self.head_size], dim=-1)
                    else:
                        query, key, value = self._gather_and_restore_pcp_qkv(query, key, value, attn_metadata)
                        output_local_padded_tokens_fa = (
                            attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size - output_padded.shape[0]
                        )
                        if output_local_padded_tokens_fa > 0:
                            output_padded = F.pad(
                                output, pad=(0, 0, 0, 0, 0, output_local_padded_tokens_fa), mode="constant", value=0
                            )

                prefill_key = key[self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded]
                prefill_value = value[self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded]
                slot_mapping = attn_metadata.slot_mapping[
                    self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded
                ]
                DeviceOperator.reshape_and_cache(
                    key=prefill_key,
                    value=prefill_value,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_mapping=slot_mapping,
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output_padded

    def _gather_and_restore_pcp_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
    ):
        """
        Gathers QKV chunks from all GPUs in the PCP group and restores the original
        sequence order for Context Parallelism (CP).
        """
        num_tokens = query.shape[0]
        num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded
        assert attn_metadata.prefill is not None and attn_metadata.prefill.pcp_metadata is not None
        pcp_padded_tokens_fla = attn_metadata.prefill.pcp_metadata.pcp_padded_tokens_fla
        qkv_fla = torch.cat(
            [query.reshape(num_tokens, -1), key.reshape(num_tokens, -1), value.reshape(num_tokens, -1)],
            dim=-1,
        )
        # The hybrid linear partitioning may result in different data on two cards, so padding is required here.
        real_num_tokens = attn_metadata.prefill.pcp_metadata.total_num_scheduled_tokens
        qkv_fla = qkv_fla[:real_num_tokens]
        if pcp_padded_tokens_fla > 0:
            qkv_fla = F.pad(qkv_fla, pad=(0, 0, 0, pcp_padded_tokens_fla), mode="constant", value=0)
        all_qkv = get_pcp_group().all_gather(
            qkv_fla[: attn_metadata.prefill.pcp_metadata.max_num_tokens_across_pcp].contiguous(), dim=0
        )
        # Restore the original sequence order using pre-computed indices
        pcp_enter_fa_restore_idx = (
            attn_metadata.prefill.pcp_metadata.pcp_enter_fa_restore_idx if attn_metadata.prefill.pcp_metadata else None
        )
        actual_qkv = torch.index_select(all_qkv, 0, pcp_enter_fa_restore_idx)
        qkv_fa_padding_workspace = query.new_empty(
            (num_actual_tokens_pcp_padded, (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
        )

        decode_offset = attn_metadata.num_decode_tokens * self.pcp_size
        qkv_fa_padding_workspace[:decode_offset] = actual_qkv[:decode_offset]

        pcp_unpad_mask = attn_metadata.prefill.pcp_metadata.pcp_unpad_mask[
            attn_metadata.num_decode_tokens * self.pcp_size :
        ]
        qkv_fa_padding_workspace[decode_offset:][pcp_unpad_mask] = actual_qkv[decode_offset:]
        qkv_fa_padding_workspace[decode_offset:][~pcp_unpad_mask] = 0

        q, k, v = qkv_fa_padding_workspace.split(
            [
                self.num_heads * self.head_size,
                self.num_kv_heads * self.head_size,
                self.num_kv_heads * self.head_size,
            ],
            dim=-1,
        )
        q_fa = query.new_empty((num_actual_tokens_pcp_padded // self.pcp_size, self.num_heads * self.head_size))
        q_fa[: attn_metadata.num_decode_tokens] = q[: decode_offset : self.pcp_size]
        pcp_fa_query_idx = attn_metadata.prefill.pcp_metadata.pcp_fa_query_idx
        q_fa[attn_metadata.num_decode_tokens :] = torch.index_select(q[decode_offset:], 0, pcp_fa_query_idx)
        return (
            q_fa.reshape(-1, self.num_heads, self.head_size),
            k.reshape(-1, self.num_kv_heads, self.head_size),
            v.reshape(-1, self.num_kv_heads, self.head_size),
        )

    def _gather_global_context_output(self, local_context_attn_output):
        if self.dcp_size > 1:
            dcp_context_attn_output = torch.empty_like(local_context_attn_output)
            dist.all_to_all_single(dcp_context_attn_output, local_context_attn_output, group=self.dcp_group)
        else:
            dcp_context_attn_output = local_context_attn_output

        if self.pcp_size > 1:
            # AllGather out&lse within CP group
            global_context_attn_output = get_pcp_group().all_gather(dcp_context_attn_output, dim=-1)
        else:
            global_context_attn_output = dcp_context_attn_output

        return global_context_attn_output

    def _update_global_context_output(self, global_context_output):
        B_total, H_total, D_plus_1 = global_context_output.shape
        S = B_total // self.pcp_size
        H = H_total // self.dcp_size
        D = self.head_size
        assert D_plus_1 == D + 1
        # [PCP, S, DCP, H, D+1]
        x = global_context_output.view(self.pcp_size, S, self.dcp_size, H, D_plus_1)
        # [PCP, DCP, S, H, D+1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Flatten [N, S, H, D+1], N = pcp_size * dcp_size
        x = x.view(-1, S, H, D_plus_1)
        # Split out lse
        attn_out_allgather, attn_lse_allgather = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
        context_output, context_lse = _update_out_and_lse(attn_out_allgather, attn_lse_allgather)
        return context_output, context_lse

>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendAttentionDCPMetadata)
        num_decode_tokens = attn_metadata.num_decode_tokens
        if attn_metadata.num_decodes > 0:
            assert attn_metadata.decode is not None and attn_metadata.decode.actual_seq_lengths_q is not None
            # Preserve upstream graph-padding queries; prefill keeps its real-token offset.
            decode_query_end = attn_metadata.decode.actual_seq_lengths_q[-1]
            output[:decode_query_end] = self._forward_decode_dcp(
                query[:decode_query_end].contiguous(),
                attn_metadata,
                key[:decode_query_end] if key is not None else None,
                value[:decode_query_end] if value is not None else None,
            )
        if attn_metadata.num_prefills == 0:
            return output

        prefill = attn_metadata.prefill
        assert prefill is not None
        chunked_context = prefill.chunked_context
        main_stream = torch.npu.current_stream()
        if chunked_context is not None:
            # PCP query gather includes padding; current attention uses actual local tokens.
            query_end = attn_metadata.num_actual_tokens
            if self.pcp_enabled:
                assert attn_metadata.pcp_local_num_input_tokens is not None
                query_end = attn_metadata.pcp_local_num_input_tokens
            history_query = query[num_decode_tokens:query_end].contiguous()
            comm_stream = cp_chunkedprefill_comm_stream()
            comm_stream.wait_stream(main_stream)
            with torch_npu.npu.stream(comm_stream):
                history_query = self._prefill_query_all_gather(attn_metadata, history_query)

        current_output, current_lse = self._forward_prefill_current_kv(query, key, value, attn_metadata)
        if chunked_context is not None:
            main_stream.wait_stream(comm_stream)
            history_output, history_lse = self._compute_prefill_context(history_query, kv_cache, attn_metadata)
            comm_stream.wait_stream(main_stream)
            with torch_npu.npu.stream(comm_stream):
                history_recv = self._merge_dcp_attention_output(history_output, history_lse, defer_combine=True)
            main_stream.wait_stream(comm_stream)
            if self.pcp_enabled:
                assert chunked_context.pcp_local_query_indices is not None
                history_recv = torch.index_select(history_recv, 2, chunked_context.pcp_local_query_indices)
            current_output = fused_dcp_lse_combine(
                history_recv,
                self.head_size,
                scatter_dim=1,
                local_output=current_output,
                local_lse=current_lse,
            )

<<<<<<< HEAD
        output[num_decode_tokens : num_decode_tokens + current_output.shape[0]] = current_output
=======
            if self.pcp_size > 1:
                # Scenario of Enabling PCP or PCP&DCP
                # prepare qkv and compute the head part // overlap the communication of all gather q
                data_head, data_tail = self._forward_prefill_cp_pre(prefill_query, key, value, attn_metadata)
                output_head, lse_head = self._forward_prefill_cp_attn(data_head, True, attn_metadata)
            else:
                # Scenario of Enabling DCP Individually
                attn_output_prefill, attn_lse_prefill = torch.ops.npu.npu_fused_infer_attention_score(
                    prefill_query,
                    key,
                    value,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    atten_mask=attn_metadata.attn_mask,
                    scale=self.scale,
                    sparse_mode=3,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,
                    actual_seq_lengths_kv=attn_metadata.prefill.actual_seq_lengths_q,
                    actual_seq_lengths=attn_metadata.prefill.actual_seq_lengths_q,
                )

            if has_chunked_context:
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                # computation of context
                context_output = self._compute_prefill_context(prefill_query_all, kv_cache, attn_metadata)
                # Note(qcs): (output, lse) -> [Seq, Head_num, Head_dim+1] -> [Head_num, Head_dim+1, Seq]
                local_context_output = torch.cat(context_output, dim=-1).permute([1, 2, 0]).contiguous()

                # all2all and all_gather output&lse // overlap the computation inner current chunk
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    global_context_output = self._gather_global_context_output(local_context_output)

            if self.pcp_size > 1:
                # compute the tail part and reorg output&lse // overlap the communication of output
                output_tail, lse_tail = self._forward_prefill_cp_attn(data_tail, False, attn_metadata)

                attn_output_prefill, attn_lse_prefill = self._forward_prefill_cp_post(
                    [output_head, output_tail],
                    [lse_head, lse_tail],
                    attn_metadata,
                )

            if has_chunked_context:
                # update the output of current chunk with context part
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                global_context_output = global_context_output.permute([2, 0, 1]).contiguous()
                context_output, context_lse = self._update_global_context_output(global_context_output)
                self._update_chunk_attn_out_lse_with_current_attn_out_lse(
                    attn_output_prefill, attn_lse_prefill, context_output, context_lse, prefill_query, attn_metadata
                )

            if self.pcp_size > 1 and pcp_use_hybrid_attn:
                # layer_idx != num_layers - 1
                assert attn_metadata.prefill.pcp_metadata is not None
                pcp_exit_fa_scatter_idx = attn_metadata.prefill.pcp_exit_fa_scatter_idx
                attn_output_prefill = get_pcp_group().all_gather(attn_output_prefill.contiguous(), dim=0)
                attn_output_prefill = torch.index_select(attn_output_prefill, 0, pcp_exit_fa_scatter_idx)

            output[num_decode_tokens : attn_output_prefill.shape[0] + num_decode_tokens] = attn_output_prefill
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
        return output
