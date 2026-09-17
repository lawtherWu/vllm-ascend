# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.cache_layout import (
    CacheComponentRole,
    build_mla_layer_cache_descriptor,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.decode_host_kv_pool import (
    DecodeHostKVPool,
    HostRegionKey,
)


class CPUAllocator:
    def empty_device_visible_host(self, shape, *, dtype, alignment):
        del alignment
        return torch.empty(shape, dtype=dtype)


def _descriptor(layer_id: int, group_id: int = 0, mtp: bool = False):
    spec = SimpleNamespace(
        block_size=4,
        num_kv_heads=1,
        sparse_head_dim=(8, 2, 4),
        dtype=torch.bfloat16,
        cache_sparse_c8=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )
    return build_mla_layer_cache_descriptor(
        layer_name=f"layer.{layer_id}",
        layer_id=layer_id,
        group_id=group_id,
        cache_family_id=f"family-{group_id}",
        kv_cache_spec=spec,
        is_mtp_layer=mtp,
    )


def test_host_pool_mirrors_blocks_and_only_allocates_main():
    descriptor = _descriptor(0)
    pool = DecodeHostKVPool(CPUAllocator())
    pool.initialize({0: 3}, [descriptor, _descriptor(1, mtp=True)])

    assert len(pool.regions) == 2
    assert len(pool.tensors_for_registration()) == 2
    key = HostRegionKey(0, "family-0", 0, CacheComponentRole.MAIN_KV)
    region = pool.regions[key]
    assert pool.resolve(key, 2) == region.base_address + 2 * region.page_stride_bytes
    assert pool.layer_views(0, "family-0", 0)[CacheComponentRole.MAIN_KV] is region.tensor
    with pytest.raises(IndexError):
        pool.resolve(key, 3)


def test_host_pool_supports_group_specific_capacities():
    pool = DecodeHostKVPool(CPUAllocator())
    pool.initialize({0: 2, 1: 5}, [_descriptor(0, 0), _descriptor(1, 1)])
    key = HostRegionKey(1, "family-1", 1, CacheComponentRole.MAIN_K_ROPE)
    assert pool.regions[key].capacity_blocks == 5


def test_host_pool_is_initialized_once():
    pool = DecodeHostKVPool(CPUAllocator())
    pool.initialize({0: 1}, [_descriptor(0)])
    with pytest.raises(RuntimeError, match="already initialized"):
        pool.initialize({0: 1}, [_descriptor(0)])
