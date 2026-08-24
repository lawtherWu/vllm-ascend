# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""GLM5.2 prefill MoE chunking.

GLM5.2 is registered by vLLM as ``glm_moe_dsa`` and uses the upstream
``DeepseekV2MoE`` implementation.  Keep this optimization in a worker patch
so the vllm-ascend plugin does not modify vLLM's model source or affect other
DeepSeek models.
"""

import math

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_config import get_ascend_config

_ORIGINAL_DEEPSEEK_V2_MOE_FORWARD = DeepseekV2MoE.forward


def _metadata_has_prefill(metadata) -> bool:
    if metadata is None:
        return False
    if isinstance(metadata, dict):
        return any(_metadata_has_prefill(value) for value in metadata.values())
    if isinstance(metadata, (list, tuple)):
        return any(_metadata_has_prefill(value) for value in metadata)
    return int(getattr(metadata, "num_prefills", 0) or 0) > 0


def _is_glm_moe_dsa() -> bool:
    try:
        vllm_config = get_current_vllm_config()
    except AssertionError:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "glm_moe_dsa"


def _get_prefill_moe_global_chunks(self, hidden_states: torch.Tensor, max_len: int) -> int:
    forward_context = get_forward_context()
    if not _metadata_has_prefill(getattr(forward_context, "attn_metadata", None)):
        return 1

    local_token_num = int(hidden_states.shape[0])
    local_num_chunks = max(1, math.ceil(local_token_num / max_len))
    cache_key = (local_token_num, max_len)
    cached = getattr(forward_context, "glm5_moe_global_chunks", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    global_num_chunks = local_num_chunks
    ep_group = getattr(self, "ep_group", None)
    ep_size = getattr(self, "ep_size", 1)
    if ep_size > 1:
        max_num_chunks = torch.tensor([local_num_chunks], dtype=torch.int32, device=hidden_states.device)
        torch.distributed.all_reduce(max_num_chunks, op=torch.distributed.ReduceOp.MAX, group=ep_group)
        global_num_chunks = int(max_num_chunks.item())

    forward_context.glm5_moe_global_chunks = (cache_key, global_num_chunks)
    return global_num_chunks


def _glm_moe_dsa_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if not _is_glm_moe_dsa():
        return _ORIGINAL_DEEPSEEK_V2_MOE_FORWARD(self, hidden_states)

    max_len = getattr(get_ascend_config(), "moe_chunk_max_len", 65536)
    if max_len <= 0:
        return _ORIGINAL_DEEPSEEK_V2_MOE_FORWARD(self, hidden_states)

    num_tokens, hidden_dim = hidden_states.shape
    original_hidden_states = hidden_states
    hidden_states = hidden_states.view(-1, hidden_dim)
    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    num_chunks = _get_prefill_moe_global_chunks(self, hidden_states, max_len)
    if num_chunks == 1:
        return _ORIGINAL_DEEPSEEK_V2_MOE_FORWARD(self, original_hidden_states)

    output_chunks = []
    for hidden_states_chunk in torch.tensor_split(hidden_states, num_chunks, dim=0):
        if self.experts.is_internal_router:
            output_chunk = self.experts(hidden_states=hidden_states_chunk, router_logits=hidden_states_chunk)
        else:
            router_logits, _ = self.gate(hidden_states_chunk)
            output_chunk = self.experts(hidden_states=hidden_states_chunk, router_logits=router_logits)
        output_chunks.append(output_chunk)

    final_hidden_states = torch.cat(output_chunks, dim=0)
    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    return final_hidden_states.view(num_tokens, hidden_dim)


DeepseekV2MoE.forward = _glm_moe_dsa_forward
