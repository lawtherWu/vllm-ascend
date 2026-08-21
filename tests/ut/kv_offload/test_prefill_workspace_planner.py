# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_config_from_groups,
    _get_group_layer_kv_cache_specs,
    _get_layerwise_decode_kv_cache_config,
    _get_layerwise_prefill_kv_cache_config,
)


def _config(*, enabled: bool = True, role: str = "kv_producer"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_connector_extra_config={
                "layerwise_host_kv_offload": enabled,
            },
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
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


def _merged_mla_group() -> tuple[KVCacheGroupSpec, AscendMLAAttentionSpec]:
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=28,
        sparse_head_dim=(16, 8, 4),
        dtype=torch.bfloat16,
        cache_sparse_c8=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )
    return KVCacheGroupSpec([
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ], spec), spec


def test_merged_mla_group_is_expanded_per_layer() -> None:
    group, merged_spec = _merged_mla_group()
    specs = _get_group_layer_kv_cache_specs(group)

    assert specs == {
        "model.layers.0.self_attn.attn": merged_spec,
        "model.layers.1.self_attn.attn": merged_spec,
    }


def test_prefill_planner_accepts_merged_ascend_mla_spec() -> None:
    group, spec = _merged_mla_group()
    result = _get_layerwise_prefill_kv_cache_config(
        _config(),
        [group],
        spec.page_size_bytes * 3,
    )

    assert result.num_blocks == 3
    assert result.kv_cache_tensors[0].shared_by == group.layer_names


@pytest.mark.parametrize("num_speculative_tokens", [None, 3])
def test_decode_planner_accepts_merged_ascend_mla_spec(
    monkeypatch,
    num_speculative_tokens,
) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, spec = _merged_mla_group()

    def fake_base(_config, groups, _available_memory):
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes, shared_by=groups[0].layer_names)
            ],
            kv_cache_groups=groups,
        )

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        fake_base,
    )
    result = _get_layerwise_decode_kv_cache_config(
        config,
        [group],
        1 << 30,
    )

    assert result.num_blocks > 0
    assert result.kv_cache_tensors[0].shared_by == group.layer_names


def test_decode_planner_accepts_zero_memory_for_minimal_profile(monkeypatch) -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.cache_config.num_gpu_blocks_override = 1
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, spec = _merged_mla_group()

    def fake_base(_config, groups, available_memory):
        assert available_memory >= spec.page_size_bytes
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes, shared_by=groups[0].layer_names)
            ],
            kv_cache_groups=groups,
        )

    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        fake_base,
    )
    result = _get_layerwise_decode_kv_cache_config(
        config,
        [group],
        0,
    )

    assert result.num_blocks == 1
    assert result.kv_cache_tensors[0].shared_by == group.layer_names


def test_decode_planner_rejects_zero_memory_without_profile_override() -> None:
    config = _config(enabled=True, role="kv_consumer")
    config.model_config.max_model_len = 1024
    config.scheduler_config = SimpleNamespace(max_num_seqs=1)
    group, _ = _merged_mla_group()

    with pytest.raises(MemoryError, match="cannot reserve fixed DSA workspace"):
        _get_layerwise_decode_kv_cache_config(config, [group], 0)


def test_group_spec_helper_rejects_unsupported_spec() -> None:
    group = KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        ),
    )
    with pytest.raises(NotImplementedError, match="merged MLAAttentionSpec"):
        _get_group_layer_kv_cache_specs(group)
