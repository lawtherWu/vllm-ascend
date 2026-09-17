#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import MoERunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _moe_forward_shared, _unpack
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    PreparedSharedExpertExecution,
    SharedExpertParallelMode,
    SubmittedSharedExpertGateUp,
)


def _ascend_moe_forward_shared_sp_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: object,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer the local SP outputs of the shared-expert custom op.

    The explicit shared input contains all gathered tokens, while the real
    shared output has already been reduce-scattered back to the routed input's
    local token count. The upstream fake uses the shared input's token count,
    which is only correct before the early all-gather optimization.
    """
    del router_logits, input_ids, layer_name
    assert shared_experts_input is not None
    shared_out = shared_experts_input.new_empty(
        (*hidden_states.shape[:-1], shared_experts_input.shape[-1]),
    )
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded),
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# Keep the gathered shared input as an explicit graph dependency, but expose
# the SP-local shared output shape to torch.compile/ACL graph tracing.
direct_register_custom_op(
    op_name="ascend_moe_forward_shared_sp",
    op_func=_moe_forward_shared,
    fake_impl=_ascend_moe_forward_shared_sp_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


<<<<<<< HEAD
class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
    def __init__(
        self,
        layer_name,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: AscendRoutedExperts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1,
    ):
        super().__init__(
=======
@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0


def mock_false():
    return False


def mock_true():
    return True


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        if not vllm_version_is("0.23.0"):
            w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2)
            layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

            w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2)
            layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
        else:
            w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
            layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

            w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
            layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        # TODO: Current dispatch_ffn_combine fusion operator ONLY supports NZ format.
        # Therefore, we must cast weights to NZ when fusion is enabled.
        # Once the underlying dispatch_ffn_combine operator is updated to support
        # ND format (or other formats), remove this specific 'if' check and the forced
        # npu_format_cast. At that point, the operator should be able to handle weights
        # in their native format without explicit casting here.
        if get_ascend_config().enable_fused_mc2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        else:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        input_ids = getattr(get_forward_context(), "input_ids", None)
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=self.tid2eid,
            input_ids=input_ids,
        )
        if vllm_version_is("0.23.0"):
            model_config = layer.vllm_config.model_config
        else:
            try:
                _vllm_config = get_current_vllm_config()
            except AssertionError:
                _vllm_config = None
            model_config = None if _vllm_config is None else _vllm_config.model_config
        if model_config is not None and model_config.enable_return_routed_experts:
            capturer = getattr(layer, "_ascend_routed_experts_capturer", None)
            if capturer is not None:
                capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_logical_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
            w2_scale = None
            w1_scale_bias = None
            w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


if vllm_version_is("0.23.0"):
    from vllm_ascend.ops.fused_moe.fused_moe_0_23_0 import AscendFusedMoE, AscendMoERunner

    AscendFusedMoE.__module__ = __name__
    AscendMoERunner.__module__ = __name__

else:

    class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
        moe_counter = -1
        gate_stream: torch.npu.Stream | None = None

        def __init__(
            self,
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
            layer_name,
            moe_config,
            router,
            routed_experts,
            enable_dbo,
            gate,
            shared_experts,
            shared_expert_gate,
            routed_input_transform,
            routed_output_transform,
            routed_scaling_factor,
        )
        self._gate = gate
        self.hidden_size = moe_config.hidden_dim

        self.quant_type = routed_experts.quant_type
        self.routed_experts.router = router

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        if self.moe_config.ep_size > 1:
            self.moe_config.ep_group = get_ep_group()
            self.moe_config.mc2_group = get_mc2_group()

        # Internal-router: precast weight_fp32 at load to avoid hot-path Cast.
        # Use ctor `gate` (not self.is_internal_router): Module.__getattr__ shadows during init.
        if gate is not None and not hasattr(gate, "weight_fp32"):
            gate.precast_fp32_weight = True

        self.ascend_shared_experts = None
        if shared_experts is not None:
            routed_experts.return_with_event = True
            self.ascend_shared_experts = AscendSharedExperts(
                shared_experts,
                self.moe_config,
                self.quant_type,
                self._quant_method,
            )
            if self._can_overlap_sp_shared_with(self.routed_input_transform):
                self._forward_entry = torch.ops.vllm.ascend_moe_forward_shared_sp

        setup_moe_comm_method(self.moe_config)
        # Communication objects are shared across MoE layers. Keep the bound
        # activation with its owning layer when a later layer replaces them.
        self.routed_experts.mega_moe_activation_kwargs = getattr(
            get_moe_comm_method(MoECommType.FUSED_MC2, self.moe_config), "mega_moe_activation_kwargs", None
        )
        alltoall_comm = get_moe_comm_method(MoECommType.ALLTOALL, self.moe_config)
        if alltoall_comm is not None:
            expert_ids_per_ep_rank = getattr(alltoall_comm.token_dispatcher, "expert_ids_per_ep_rank", None)
            if expert_ids_per_ep_rank is not None:
                self.routed_experts.register_buffer(
                    "expert_ids_per_ep_rank",
                    expert_ids_per_ep_rank,
                    persistent=False,
                )

<<<<<<< HEAD
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mirror of upstream ``MoERunner.forward`` (this class is instantiated
        through the OOT PluggableLayer dispatch of ``MoERunner``).
=======
            self.quant_type = self._get_quant_type()
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)

        Upstream inlines the shared+routed output combine, which is not
        exposed as an overridable hook. On NPU the combine goes through the
        fused multi-tensor add kernel (``torch._foreach_add``), whose kernel
        launch count is independent of the tensor-list length and which is
        the NPU-validated fast path for the MoE combine. Every other step,
        including the runtime-aware reduction hooks (#16550), is inherited
        unchanged.

        Future plan: once the aclnnAdd_AddAiCore_Add kernel is available via
        the superkernel integration, upstream's plain add becomes the fast
        path and this override can be dropped.
        """
        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = self._maybe_reduce_routed_output_before_transform(
            fused_output,
            fused_output_is_reduced,
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output, fused_output_is_reduced)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = torch._foreach_add([shared_output], [fused_output])[0]
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform, fused_output_is_reduced)

        return self._maybe_add_zero_expert_output(result)

    @property
    def is_internal_router(self) -> bool:
        # main (cdc4824a21): vllm#51838 removed the gate branch in
        # DeepseekV2MoE.forward, always passing router_logits=hidden_states.
        # The runner must recompute router_logits via the gate.
        return self.gate is not None

    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    def _can_overlap_sp_shared_with(self, routed_transform: object | None) -> bool:
        """Limit SP shared-expert overlap changes to routed transforms.

        Kimi-K3 is currently the only Ascend model that supplies these latent
        transforms. Keep the guard capability-based so every model without
        them stays on the original single-stream synchronization path.
        """
        shared_experts = getattr(self, "ascend_shared_experts", None)
        return (
            routed_transform is not None
            and shared_experts is not None
            and shared_experts.multistream_overlap
            and shared_experts.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        )

    @property
    def _fused_output_is_reduced(self) -> bool:
        # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
        # TP all-reduce for the routed output, and AscendSharedExperts.forward
        # handles it for the shared output. Signal this to the upstream
        # MoERunner.forward() so _maybe_reduce_final_output does not apply a
        # second TP all-reduce (which would double-count the contributions).
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and self.moe_config.is_sequence_parallel)

    def _get_shared_expert_parallel_mode(self) -> SharedExpertParallelMode:
        shared_experts = getattr(self, "ascend_shared_experts", None)
        if shared_experts is None or not hasattr(shared_experts, "parallel_mode"):
            return SharedExpertParallelMode.TENSOR_PARALLEL
        return shared_experts.parallel_mode()

    @property
    def local_num_experts(self) -> int:
        """Number of physical experts managed by this EPLB layer."""
        return self.moe_config.num_local_experts

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

<<<<<<< HEAD
    def _should_reduce_routed_before_combine(self) -> bool:
        """Static layer policy; communication-dependent decisions stay in the op."""
        # Shared DP already produces a complete output, so reduce routed first.
        if self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            return True
        # A routed transform must receive the complete TP result.
        if self.routed_output_transform is None or self.moe_config.is_sequence_parallel:
=======
        def _get_quant_type(self) -> QuantType:
            quant_type = QuantType.NONE
            method = getattr(self._quant_method, "quant_method", None)

            if method is not None:
                quant_type = getattr(method, "quant_type", QuantType.NONE)

            return quant_type

        @property
        def is_internal_router(self) -> bool:
            gate = self.gate
            return gate is not None and hasattr(gate, "weight_fp32")

        @property
        def use_dp_chunking(self) -> bool:
            """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
            chunked path. Always return False to stay on forward_impl."""
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
            return False
        return self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1

    # Shared-expert layout-specific communication is handled by
    # AscendSharedExperts, so only standard TP weights need a separate
    # all-reduce when routed output has already been reduced.
    def _maybe_reduce_shared_expert_output(  # type: ignore[misc]
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        if (
            shared_output is None
            or self._get_shared_expert_parallel_mode() is not SharedExpertParallelMode.TENSOR_PARALLEL
        ):
            return shared_output
        # The upstream boolean can be specialized during tracing. Only the
        # model-static early-reduction policy may be tested outside the op.
        if self._should_reduce_routed_before_combine():
            return tensor_model_parallel_all_reduce(shared_output)
        return torch.ops.vllm.maybe_all_reduce_shared_expert(shared_output, self.layer_name)

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        if self._should_reduce_routed_before_combine():
            fused_output = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(fused_output, self.layer_name)
            return fused_output, True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(  # type: ignore[misc]
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        # Do not branch on output_is_reduced: it can describe the tracing
        # batch rather than the batch replaying this graph. Early reduction
        # and sequence parallelism are static properties of the layer.
        if not self.moe_config.is_sequence_parallel and not self._should_reduce_routed_before_combine():
            states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states, self.layer_name)
        if trunc_size is not None and trunc_size > 0:
            return states[..., :trunc_size]
        return states

<<<<<<< HEAD
    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context
        if self.ascend_shared_experts is not None:
            self.ascend_shared_experts.set_lora_context(lora_context)

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Start the SP shared-input gather before the latent down projection."""
        prepared_input = PreparedSharedExpertInput(hidden_states)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_input_transform):
            assert shared_experts is not None
            prepared_input = shared_experts.prepare_input_async(hidden_states)
        routed_input, shared_input = super().apply_routed_input_transform(hidden_states)
        if prepared_input.ready_event is not None:
            torch.npu.current_stream().wait_event(prepared_input.ready_event)
            shared_input = prepared_input.hidden_states
        return routed_input, shared_input
=======
        def no_shared_forward_impl(  # type: ignore[override]
            self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
        ) -> torch.Tensor | FusedMoEResult:
            forward_context = get_forward_context()
            # When static kernels are enabled, the forward pass runs twice (compilation + capture),
            # causing moe_layer_index to overflow. Wrap the index to prevent out-of-bounds errors.
            if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
                moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
                forward_context.moe_layer_index = moe_layer_index
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)

    def apply_routed_output_transform(self, fused_output: torch.Tensor) -> torch.Tensor:
        """Run the latent up projection before joining the SP shared output."""
        fused_output = super().apply_routed_output_transform(fused_output)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_output_transform):
            assert shared_experts is not None
            shared_experts.wait_for_output()
        return fused_output

    def _prepare_router_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare only the input needed by a pre-cast FP32 router gate."""
        gate = self.gate
        assert gate is not None
        if not hasattr(gate, "weight_fp32"):
            return hidden_states
        return router_logits if router_logits.dtype == torch.float32 else hidden_states.float()

    def _apply_router_linear(self, router_input: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the internal router gate to an already prepared input."""
        gate = self.gate
        assert gate is not None
        weight = getattr(gate, "weight", None)
        weight_fp32 = getattr(gate, "weight_fp32", None)
        if weight is None and weight_fp32 is None:
            gate_out = gate(router_input)
            return gate_out[0] if isinstance(gate_out, tuple) else gate_out
        # The device adaptor fuses the bf16->fp32 cast into the gate matmul on
        # A5 (npu_fused_matmul "16cast32"); other devices reuse the prepared
        # FP32 input with the pre-cast FP32 weight.
        return DeviceOperator.compute_gate_logits(
            hidden_states,
            weight,
            weight_fp32,
            prepared_input=router_input,
        )

    def _compute_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute router logits without extending a fresh FP32 cast's lifetime.

        Gates without a pre-cast weight retain their registered forward path
        and do not allocate an unused FP32 input.
        """
        router_input = self._prepare_router_input(hidden_states, router_logits)
        return self._apply_router_linear(router_input, hidden_states)

    def _prepare_shared_for_router_overlap(
        self,
        prepared_shared_input: PreparedSharedExpertInput,
        shared_input_ready: torch.npu.Event,
    ) -> PreparedSharedExpertExecution | None:
        shared_experts = self.ascend_shared_experts
        assert shared_experts is not None
        if not getattr(shared_experts, "multistream_overlap", False):
            return None
        return shared_experts.prepare_for_router_overlap(
            prepared_shared_input,
            shared_input_ready,
        )

    def _prepare_router_and_milestones(
        self,
        prepared_shared_input: PreparedSharedExpertInput,
        shared_hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.npu.Event,
        torch.npu.Event,
        SubmittedSharedExpertGateUp | None,
    ]:
        if self.is_internal_router:
            router_input = self._prepare_router_input(shared_hidden_states, router_logits)
            # Release shared Vector/AIV preparation after any external FP32
            # router-input Cast. Gates using their registered forward path may
            # still prepare their input internally.
            shared_input_ready = torch.npu.current_stream().record_event()
            prepared_execution = self._prepare_shared_for_router_overlap(
                prepared_shared_input,
                shared_input_ready,
            )
            router_logits = self._apply_router_linear(router_input, shared_hidden_states)
            del router_input
            router_output_ready = torch.npu.current_stream().record_event()
        else:
            shared_input_ready = torch.npu.current_stream().record_event()
            prepared_execution = self._prepare_shared_for_router_overlap(
                prepared_shared_input,
                shared_input_ready,
            )
            router_output_ready = shared_input_ready
        # Submit Gate-Up now, before routed_experts.forward_impl enqueues TopK
        # and dispatch. The auxiliary stream waits for router_output_ready, so
        # Gate-Up starts together with routed TopK and can cover dispatch.
        shared_experts = self.ascend_shared_experts
        assert shared_experts is not None
        submitted_gate_up = (
            shared_experts.enqueue_gate_up_after_router(
                prepared_execution,
                router_output_ready,
            )
            if prepared_execution is not None
            else None
        )
        return router_logits, shared_input_ready, router_output_ready, submitted_gate_up

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self._sequence_parallel_context():
            shared_hidden_states = shared_experts_input if shared_experts_input is not None else hidden_states
            if self.ascend_shared_experts is None:
                if self.is_internal_router:
                    router_logits = self._compute_router_logits(hidden_states, router_logits)
                return self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
            shared_input_is_gathered = self._can_overlap_sp_shared_with(self.routed_input_transform)
            defer_shared_output_wait = self._can_overlap_sp_shared_with(self.routed_output_transform)
            prepared_shared_input = (
                PreparedSharedExpertInput(shared_hidden_states, is_gathered=True)
                if shared_input_is_gathered
                else self.ascend_shared_experts.prepare_input_before_routed(shared_hidden_states)
            )
            router_logits, shared_input_ready, router_output_ready, submitted_gate_up = (
                self._prepare_router_and_milestones(
                    prepared_shared_input,
                    shared_hidden_states,
                    router_logits,
                )
            )
            routed_out, milestones = self.routed_experts.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
            assert isinstance(milestones, RoutedMoEMilestones)
            milestones.shared_input_ready = shared_input_ready
            milestones.router_output_ready = router_output_ready
            if prepared_shared_input.is_gathered:
                milestones.routed_finalize_done = torch.npu.current_stream().record_event()

            if submitted_gate_up is None:
                shared_out = self.ascend_shared_experts.forward(
                    prepared_shared_input,
                    milestones,
                    defer_output_wait=defer_shared_output_wait,
                )
            else:
                shared_out = self.ascend_shared_experts.forward(
                    prepared_shared_input,
                    milestones,
                    defer_output_wait=defer_shared_output_wait,
                    submitted_gate_up=submitted_gate_up,
                )
            return shared_out, routed_out
