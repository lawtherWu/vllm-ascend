# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.attention.cache_layout import (
    CacheComponentRole,
    build_mla_layer_cache_descriptor,
)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_config_from_groups = (
    vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
)


def is_layerwise_host_offload_prefill(vllm_config: VllmConfig) -> bool:
    """Return whether this worker uses the Prefill shared-workspace plan."""

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_role not in {
        "kv_producer",
        "kv_both",
    }:
        return False
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    return bool(extra_config.get("layerwise_host_kv_offload", False))


def _is_mtp_cache_layer(
    vllm_config: VllmConfig,
    group: KVCacheGroupSpec,
    layer_name: str,
) -> bool:
    """Identify draft layers without relying on a GLM-specific name prefix."""

    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_text_config", None)
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if isinstance(num_hidden_layers, int):
        try:
            from vllm.model_executor.models.utils import extract_layer_index

            if extract_layer_index(layer_name) >= num_hidden_layers:
                return True
        except (AssertionError, ValueError):
            pass
    lowered = layer_name.lower()
    if "mtp" in lowered or "draft" in lowered:
        return True
    return bool(group.is_eagle_group and len(group.layer_names) == 1)


def _get_layerwise_prefill_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Plan one shared Main/Indexer bundle plus ordinary MTP caches.

    The scheduler still owns the full logical block table.  ``shared_by`` only
    aliases the physical tensor used by compatible base-model layers, so HBM
    capacity is independent of their count.  Draft layers remain ordinary
    per-layer PA caches and are included in the per-block denominator.
    """

    if len(kv_cache_groups) != 1:
        raise NotImplementedError(
            "Layerwise Host Offload Prefill currently requires one uniform "
            "GLM cache group"
        )
    group = kv_cache_groups[0]
    if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        raise NotImplementedError(
            "Layerwise Host Offload Prefill requires UniformTypeKVCacheSpecs"
        )

    per_layer_specs = group.kv_cache_spec.kv_cache_specs
    workspace_layers: list[str] = []
    mtp_layers: list[str] = []
    for layer_name in group.layer_names:
        if _is_mtp_cache_layer(vllm_config, group, layer_name):
            mtp_layers.append(layer_name)
        else:
            workspace_layers.append(layer_name)
    if not workspace_layers:
        raise ValueError("Layerwise Host Offload Prefill found no base-model layers")

    workspace_page_sizes = {
        per_layer_specs[layer_name].page_size_bytes
        for layer_name in workspace_layers
    }
    if len(workspace_page_sizes) != 1:
        raise NotImplementedError(
            "Layerwise Host Offload Prefill requires compatible base-layer "
            "cache page sizes; heterogeneous layers need separate arenas"
        )
    workspace_page_size = next(iter(workspace_page_sizes))
    mtp_page_sizes = {
        layer_name: per_layer_specs[layer_name].page_size_bytes
        for layer_name in mtp_layers
    }
    bytes_per_block = workspace_page_size + sum(mtp_page_sizes.values())
    if bytes_per_block <= 0:
        raise ValueError("Layerwise Host Offload Prefill has an invalid page budget")

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    if num_blocks <= 0:
        raise ValueError(
            "No KV cache block fits the Layerwise Host Offload Prefill budget"
        )

    kv_cache_tensors = [
        KVCacheTensor(
            size=workspace_page_size * num_blocks,
            shared_by=workspace_layers,
        )
    ]
    kv_cache_tensors.extend(
        KVCacheTensor(
            size=mtp_page_sizes[layer_name] * num_blocks,
            shared_by=[layer_name],
        )
        for layer_name in mtp_layers
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def is_layerwise_host_offload_decode(vllm_config: VllmConfig) -> bool:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_role not in {"kv_consumer", "kv_both"}:
        return False
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    return bool(extra_config.get("layerwise_host_kv_offload", False))


def _dsa_group_count(vllm_config: VllmConfig, num_hidden_layers: int) -> int:
    """Return the number of DSA metadata sets allocated by Model Runner."""
    hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
    indexer_types = getattr(hf_config, "indexer_types", None)
    if indexer_types is None:
        return num_hidden_layers
    if len(indexer_types) < num_hidden_layers:
        raise ValueError("indexer_types is shorter than num_hidden_layers")
    groups = 0
    has_owner = False
    for indexer_type in indexer_types[:num_hidden_layers]:
        normalized = str(indexer_type).lower()
        if normalized == "full":
            groups += 1
            has_owner = True
        elif normalized == "shared":
            if not has_owner:
                raise ValueError("A shared DSA layer cannot precede its owner")
        else:
            raise NotImplementedError(
                f"Unsupported DSA indexer type: {indexer_type!r}"
            )
    if groups == 0:
        raise ValueError("No DSA metadata groups were produced")
    return groups


def _dsa_workspace_bytes(
    vllm_config: VllmConfig,
    group: KVCacheGroupSpec,
    specs: dict[str, MLAAttentionSpec],
) -> int:
    """Return the fixed Decode DSA workspace reservation."""
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
    default_pool_size = (
        16384 if bool(getattr(hf_config, "enlarge_pool_size", False)) else 8192
    )
    pool_size = int(extra_config.get("dsa_pool_size", default_pool_size))
    id_range = max(
        int(vllm_config.model_config.max_model_len),
        int(extra_config.get("dsa_id_range", 131072)),
    )
    batch_capacity = int(vllm_config.scheduler_config.max_num_seqs)
    num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError("DSA workspace accounting requires num_hidden_layers")
    metadata_groups = _dsa_group_count(vllm_config, num_hidden_layers)
    raw_seq, topk, block_size = 4, 2048, 128
    if pool_size <= 0 or pool_size > 16384 or pool_size % 16 != 0:
        raise ValueError("dsa_pool_size must be in (0, 16384] and divisible by 16")
    selection_rows = batch_capacity * raw_seq
    selection_blocks = selection_rows * (topk // block_size)
    total = 0
    dsa_layers = [
        name
        for name in group.layer_names
        if not _is_mtp_cache_layer(vllm_config, group, name)
    ]
    if not dsa_layers:
        raise ValueError("Decode Host offload found no base DSA layers")
    for layer_name in dsa_layers:
        spec = specs[layer_name]
        if (
            not isinstance(spec, MLAAttentionSpec)
            or getattr(spec, "sparse_head_dim", None) is None
            or bool(getattr(spec, "cache_sparse_c8", False))
        ):
            raise NotImplementedError(
                f"Unsupported DSA cache layout for layer {layer_name}; "
                "P0 requires non-C8 SFA"
            )
        descriptor = build_mla_layer_cache_descriptor(
            layer_name=layer_name,
            layer_id=0,
            group_id=0,
            cache_family_id="default",
            kv_cache_spec=spec,
        )
        main_bytes_per_token = sum(
            c.page_bytes // block_size for c in descriptor.main_components
        )
        indexer_bytes_per_token = sum(
            c.page_bytes // block_size for c in descriptor.indexer_components
        )
        if main_bytes_per_token <= 0 or indexer_bytes_per_token <= 0:
            raise NotImplementedError(
                f"Unsupported DSA cache layout for layer {layer_name}: "
                "Main/Indexer tuple is incomplete"
            )
        total += batch_capacity * pool_size * main_bytes_per_token
        total += selection_blocks * block_size * main_bytes_per_token
        total += selection_rows * 4
        # selection_default_indices is an expanded view of one [1, 1, topk]
        # arange allocation in Model Runner; count backing storage, not its
        # logical expanded numel.
        total += topk * 4
        total += selection_rows * (topk // block_size) * 4
    # DsaGroupMetadata is allocated once per full/shared indexer group.
    total += metadata_groups * (
        batch_capacity * pool_size * 4
        + batch_capacity * id_range * 4
        + batch_capacity * (pool_size // 16) * 4
    )
    return total


def _get_layerwise_decode_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if len(kv_cache_groups) != 1:
        raise NotImplementedError(
            "Decode Layerwise Host offload requires exactly one KV cache group; "
            "hybrid groups are not supported in the first release"
        )
    if not isinstance(kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        raise NotImplementedError("Decode Layerwise Host offload requires UniformTypeKVCacheSpecs")
    group = kv_cache_groups[0]
    if group.kv_cache_spec.block_size != 128:
        raise NotImplementedError(
            "Decode Layerwise Host offload requires block_size=128 for the "
            "recipes DsaServe PA block table"
        )
    specs = group.kv_cache_spec.kv_cache_specs
    hbm_bytes_per_block = 0
    for layer_name in group.layer_names:
        spec = specs[layer_name]
        if _is_mtp_cache_layer(vllm_config, group, layer_name):
            hbm_bytes_per_block += spec.page_size_bytes
            continue
        # The tuple differs by device/layout: A3 non-C8 has one indexer
        # component, A3 C8 has indexer K + scale, and A5 C8 has a merged Main
        # component followed by indexer K + scale. Counting only ratio[2]
        # underestimates A3-C8/A5 and can make the planner over-allocate HBM.
        # Keep this calculation in the same descriptor builder used by the
        # AttentionBackend so model-specific layout rules do not leak into the
        # planner.
        if (
            not isinstance(spec, MLAAttentionSpec)
            or getattr(spec, "sparse_head_dim", None) is None
            or bool(getattr(spec, "cache_sparse_c8", False))
        ):
            raise NotImplementedError(
                f"Decode Layerwise Host offload does not support cache layout "
                f"for layer {layer_name}; P0 requires non-C8 SFA"
            )
        descriptor = build_mla_layer_cache_descriptor(
            layer_name=layer_name,
            layer_id=0,
            group_id=0,
            cache_family_id="default",
            kv_cache_spec=spec,
        )
        hbm_bytes_per_block += sum(
            component.page_bytes
            for component in descriptor.components
            if component.component_role
            in (CacheComponentRole.INDEXER_K, CacheComponentRole.INDEXER_SCALE)
        )
    if hbm_bytes_per_block <= 0:
        raise ValueError("Decode Layerwise Host offload has no device Indexer cache")
    dsa_bytes = _dsa_workspace_bytes(vllm_config, group, specs)
    effective_memory = available_memory - dsa_bytes
    if effective_memory <= 0:
        raise MemoryError(
            "Decode Layerwise Host offload cannot reserve fixed DSA workspace "
            f"({dsa_bytes} bytes) from {available_memory} bytes"
        )
    num_blocks = may_override_num_blocks(vllm_config, effective_memory // hbm_bytes_per_block)
    if num_blocks <= 0:
        raise ValueError("No KV cache block fits Decode Indexer-only HBM budget")
    if num_blocks * hbm_bytes_per_block > effective_memory:
        raise MemoryError(
            "Configured KV cache block override exceeds Decode HBM budget after "
            f"DSA workspace reservation ({dsa_bytes} bytes)"
        )
    base = _orig_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        max(effective_memory, group.kv_cache_spec.page_size_bytes),
    )
    old_blocks = base.num_blocks
    tensors = [
        KVCacheTensor(size=(tensor.size // old_blocks) * num_blocks, shared_by=tensor.shared_by)
        for tensor in base.kv_cache_tensors
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=kv_cache_groups)


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if is_layerwise_host_offload_decode(vllm_config):
        return _get_layerwise_decode_kv_cache_config(
            vllm_config, kv_cache_groups, available_memory
        )
    if is_layerwise_host_offload_prefill(vllm_config):
        return _get_layerwise_prefill_kv_cache_config(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )
    return _orig_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the MLA groups. DSpark PD may introduce
    # transferable SWA cache pages larger than the regular MLA buckets; keep
    # those pages as their own buckets instead of rejecting the configuration.
    base_page_sizes = sorted(set(full_mla_spec.get_page_sizes()) | set(full_mla_c128_spec.get_page_sizes()))
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        oversized_page_sizes = {ps for ps in sm_page_sizes if ps > max(base_page_sizes)}
        candidate_page_sizes = sorted(set(base_page_sizes) | oversized_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in candidate_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        layer_counts_per_size = {len(layers) for layers in layers_per_size.values()}
        if len(layer_counts_per_size) != 1:
            # DSpark PD can mix regular SWA layers with transferable DSpark
            # SWA layers in the same UniformKV group. Their page buckets may
            # have different layer counts, but the draft model needs these
            # layers to share one block table. Keep the whole group intact and
            # let the final KV tensor layout handle uneven page buckets.
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=list(sm_spec.kv_cache_specs.keys()),
                    kv_cache_spec=sm_spec,
                )
            )
            continue
        num_layers_per_size = next(iter(layer_counts_per_size))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: non-full-MLA groups must have been page_size-padded
    upstream (see _get_kv_cache_groups_uniform_groups) so every layer's
    page_size matches one of the planned bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    page_sizes: list[int] = sorted(
        {page_size for group in kv_cache_groups for page_size in group.kv_cache_spec.get_page_sizes()}
    )
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = (
    _ascend_get_kv_cache_config_from_groups
)
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vllm v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
if vllm_version_is("0.23.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
else:
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
