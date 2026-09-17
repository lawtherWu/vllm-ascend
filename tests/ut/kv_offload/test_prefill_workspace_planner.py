# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_config_from_groups,
    _get_layerwise_prefill_kv_cache_config,
)


def _config(*, enabled: bool = True, role: str = "kv_producer"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, block_size=128),
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_connector_extra_config={
                "layerwise_host_kv_offload": enabled,
            },
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=2, index_topk=2048),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )


def _group(*, second_head_size: int = 16):
    specs = {
        "model.layers.0.self_attn.attn": FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        ),
        "model.layers.1.self_attn.attn": FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=second_head_size,
            dtype=torch.float16,
        ),
        "model.layers.2.self_attn.attn": FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        ),
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    return KVCacheGroupSpec(list(specs), uniform)


def test_prefill_planner_shares_one_base_bundle_and_counts_mtp() -> None:
    group = _group()
    page_size = group.kv_cache_spec.kv_cache_specs[
        "model.layers.0.self_attn.attn"
    ].page_size_bytes
    available_memory = page_size * 2 * 17

    result = _get_layerwise_prefill_kv_cache_config(
        _config(),
        [group],
        available_memory,
    )

    assert result.num_blocks == 17
    assert len(result.kv_cache_tensors) == 2
    assert result.kv_cache_tensors[0].shared_by == [
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ]
    assert result.kv_cache_tensors[1].shared_by == [
        "model.layers.2.self_attn.attn"
    ]
    assert sum(item.size for item in result.kv_cache_tensors) == available_memory


def test_prefill_planner_rejects_incompatible_workspace_layers() -> None:
    with pytest.raises(NotImplementedError, match="compatible base-layer"):
        _get_layerwise_prefill_kv_cache_config(
            _config(),
            [_group(second_head_size=32)],
            1 << 30,
        )


def test_feature_gate_delegates_when_disabled(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils."
        "_orig_get_kv_cache_config_from_groups",
        lambda *_args: sentinel,
    )
    assert (
        _ascend_get_kv_cache_config_from_groups(
            _config(enabled=False),
            [_group()],
            1 << 30,
        )
        is sentinel
    )
