#
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
#
<<<<<<< HEAD
# MiniMax-M2 on Ascend: fused attention.
=======
# MiniMax-M2 on Ascend: MoE all_reduce, fused attention, fp8 load dequant.
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
#

import torch
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Attention

from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice

<<<<<<< HEAD
=======
FP8_DTYPES = tuple(
    getattr(torch, dtype_name)
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, dtype_name)
)


# ---------------------------------------------------------------------------
# MiniMaxM2MoE.forward: use maybe_all_reduce_tensor_model_parallel
# ---------------------------------------------------------------------------
def _patched_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states.to(torch.float32))
    final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_dim)


MiniMaxM2MoE.forward = _patched_moe_forward

>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)

# ---------------------------------------------------------------------------
# MiniMaxM2Attention: fused qkv split, rmsnorm, and rope on NPU.
# ---------------------------------------------------------------------------
def _patch_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    cos, sin = get_cos_and_sin_slice()
    q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv,
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        q_hidden_size=self.q_size,
        kv_hidden_size=self.kv_size,
        head_dim=self.head_dim,
        rotary_dim=getattr(self.rotary_emb, "rotary_dim", self.head_dim),
        eps=self.q_norm.variance_epsilon,
        tp_world=self.q_norm.tp_world,
        cos=cos,
        sin=sin,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


MiniMaxM2Attention.forward = _patch_forward
