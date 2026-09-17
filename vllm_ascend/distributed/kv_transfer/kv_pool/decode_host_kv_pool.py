# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm_ascend.attention.cache_layout import (
    CacheComponentDescriptor,
    CacheComponentRole,
    LayerCacheLayoutDescriptor,
    LayerCacheViews,
    MemoryKind,
)


class HostBackingAllocator(Protocol):
    def empty_device_visible_host(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        alignment: int,
    ) -> torch.Tensor: ...


class TorchNpuSwappedMemoryAllocator:
    def __init__(self, device: torch.device | str):
        self.device = device

    def empty_device_visible_host(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        alignment: int,
    ) -> torch.Tensor:
        del alignment  # The torch_npu allocator owns the physical alignment.
        import torch_npu

        return torch_npu.empty_with_swapped_memory(
            shape,
            dtype=dtype,
            device=self.device,
        )


@dataclass(frozen=True)
class HostRegionKey:
    group_id: int
    cache_family_id: str
    layer_id: int
    component_role: CacheComponentRole


@dataclass(frozen=True)
class PhysicalHostRegion:
    tensor: torch.Tensor
    base_address: int
    capacity_blocks: int
    page_stride_bytes: int


class DecodeHostKVPool:
    """Fixed Main-KV backing that mirrors scheduler physical block IDs.

    This class deliberately has no request allocation, eviction, readiness or
    release state.  The scheduler remains the only block allocator.
    """

    def __init__(self, backing_allocator: HostBackingAllocator | None):
        self._allocator = backing_allocator
        self._regions: dict[HostRegionKey, PhysicalHostRegion] = {}
        self._initialized = False

    def initialize(
        self,
        num_blocks_by_group: Mapping[int, int],
        descriptors: Sequence[LayerCacheLayoutDescriptor],
    ) -> None:
        if self._initialized:
            raise RuntimeError("DecodeHostKVPool is already initialized")
        for descriptor in descriptors:
            if descriptor.is_mtp_layer:
                continue
            try:
                num_blocks = num_blocks_by_group[descriptor.group_id]
            except KeyError as error:
                raise ValueError(
                    f"Missing num_blocks for cache group {descriptor.group_id}"
                ) from error
            if num_blocks <= 0:
                raise ValueError("Host pool num_blocks must be positive")
            for component in descriptor.main_components:
                if component.decode_memory_kind is not MemoryKind.HOST_DEVICE_VISIBLE:
                    continue
                key = HostRegionKey(
                    group_id=descriptor.group_id,
                    cache_family_id=descriptor.cache_family_id,
                    layer_id=descriptor.layer_id,
                    component_role=component.component_role,
                )
                if key in self._regions:
                    raise ValueError(f"Duplicate Host KV region {key!r}")
                tensor = self._allocate_component(component, num_blocks)
                self._regions[key] = PhysicalHostRegion(
                    tensor=tensor,
                    base_address=tensor.data_ptr(),
                    capacity_blocks=num_blocks,
                    page_stride_bytes=component.page_stride_bytes,
                )
        if not self._regions:
            raise ValueError("DecodeHostKVPool has no Host Main components")
        self._initialized = True

    def _allocate_component(
        self,
        component: CacheComponentDescriptor,
        num_blocks: int,
    ) -> torch.Tensor:
        if self._allocator is None:
            raise RuntimeError(
                "DecodeHostKVPool requires a backing allocator when initialize() "
                "is used; adopt_existing() does not allocate"
            )
        tensor = self._allocator.empty_device_visible_host(
            (num_blocks, *component.shape_per_block),
            dtype=component.dtype,
            alignment=component.alignment_bytes,
        )
        if tensor.dtype != component.dtype:
            raise TypeError(
                f"Host allocator returned {tensor.dtype}, expected {component.dtype}"
            )
        if tuple(tensor.shape) != (num_blocks, *component.shape_per_block):
            raise ValueError("Host allocator returned an incompatible shape")
        if tensor.stride(0) * tensor.element_size() != component.page_stride_bytes:
            raise ValueError("Host allocator returned an incompatible page stride")
        return tensor

    def adopt_existing(
        self,
        num_blocks_by_group: Mapping[int, int],
        descriptors: Sequence[LayerCacheLayoutDescriptor],
        layer_caches: Mapping[str, Sequence[torch.Tensor] | torch.Tensor],
    ) -> None:
        if self._initialized:
            raise RuntimeError("DecodeHostKVPool is already initialized")
        for descriptor in descriptors:
            if descriptor.is_mtp_layer:
                continue
            num_blocks = num_blocks_by_group.get(descriptor.group_id)
            if num_blocks is None or num_blocks <= 0:
                raise ValueError(f"Missing Host KV capacity for group {descriptor.group_id}")
            if descriptor.layer_name not in layer_caches:
                raise KeyError(descriptor.layer_name)
            raw = layer_caches[descriptor.layer_name]
            raw_tuple = (raw,) if isinstance(raw, torch.Tensor) else tuple(raw)
            for component in descriptor.main_components:
                if component.decode_memory_kind is not MemoryKind.HOST_DEVICE_VISIBLE:
                    continue
                if component.tensor_index >= len(raw_tuple):
                    raise ValueError(
                        f"Host cache {descriptor.layer_name} is missing tuple component "
                        f"{component.tensor_index}"
                    )
                tensor = raw_tuple[component.tensor_index]
                if tensor.dtype != component.dtype or tensor.shape[0] != num_blocks:
                    raise ValueError(
                        f"Host cache {descriptor.layer_name}/{component.component_role.value} has incompatible shape or dtype"
                    )
                page_stride = tensor.stride(0) * tensor.element_size()
                if page_stride != component.page_stride_bytes:
                    raise ValueError(
                        f"Host cache {descriptor.layer_name}/{component.component_role.value} "
                        "has an incompatible page stride"
                    )
                key = HostRegionKey(descriptor.group_id, descriptor.cache_family_id, descriptor.layer_id, component.component_role)
                self._regions[key] = PhysicalHostRegion(tensor, tensor.data_ptr(), num_blocks, page_stride)
        if not self._regions:
            raise ValueError("DecodeHostKVPool has no Host Main components")
        self._initialized = True

    def resolve(
        self,
        key: HostRegionKey,
        block_id: int,
        in_page_offset: int = 0,
    ) -> int:
        region = self._regions[key]
        if block_id < 0 or block_id >= region.capacity_blocks:
            raise IndexError(
                f"block_id {block_id} is outside [0, {region.capacity_blocks})"
            )
        if in_page_offset < 0 or in_page_offset >= region.page_stride_bytes:
            raise IndexError(
                f"in_page_offset {in_page_offset} is outside page stride "
                f"{region.page_stride_bytes}"
            )
        return (
            region.base_address
            + block_id * region.page_stride_bytes
            + in_page_offset
        )

    def layer_views(
        self,
        group_id: int,
        cache_family_id: str,
        layer_id: int,
    ) -> LayerCacheViews:
        views = {
            key.component_role: region.tensor
            for key, region in self._regions.items()
            if key.group_id == group_id
            and key.cache_family_id == cache_family_id
            and key.layer_id == layer_id
        }
        if not views:
            raise KeyError((group_id, cache_family_id, layer_id))
        return LayerCacheViews(views)

    def tensors_for_registration(self) -> tuple[torch.Tensor, ...]:
        return tuple(region.tensor for region in self._regions.values())

    @property
    def regions(self) -> Mapping[HostRegionKey, PhysicalHostRegion]:
        return self._regions
