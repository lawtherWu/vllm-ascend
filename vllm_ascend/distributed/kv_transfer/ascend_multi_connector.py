from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import (
    LayerWorkspaceFence,
    WorkspaceKey,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = {} if kv_transfer_config is None else (
            kv_transfer_config.kv_connector_extra_config or {}
        )
        self._layerwise_workspace_fences_enabled = bool(
            extra_config.get("layerwise_host_kv_offload", False)
        ) and bool(kv_transfer_config and kv_transfer_config.is_kv_producer)
        self._execution_id = -1
        self._layer_id = 0
        self._workspace_generation = 0
        self._active_workspace_fence: LayerWorkspaceFence | None = None
        self._workspace_arena_id = 0
        if self._layerwise_workspace_fences_enabled:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
                AscendStoreConnector,
            )

            store_children = [
                connector
                for connector in self._connectors
                if isinstance(connector, AscendStoreConnector)
            ]
            p2p_children = [
                connector
                for connector in self._connectors
                if isinstance(connector, MooncakeLayerwiseConnector)
            ]
            if not store_children or not p2p_children:
                raise ValueError(
                    "Prefill layerwise host offload requires both an "
                    "AscendStoreConnector child and a "
                    "MooncakeLayerwiseConnector child"
                )
            if any(
                not getattr(connector, "use_layerwise", False)
                or not getattr(connector, "use_layerwise_range", False)
                for connector in store_children
            ):
                raise ValueError(
                    "Prefill layerwise host offload requires the Store child "
                    "configuration use_layerwise=true and "
                    "use_layerwise_range=true"
                )

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            if i == chosen_connector or isinstance(connector, MooncakeLayerwiseConnector):
                connector.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                connector.update_state_after_alloc(request, empty_blocks, 0)

    def _wait_for_active_workspace(self) -> None:
        fence = self._active_workspace_fence
        if fence is None:
            return
        self._active_workspace_fence = None
        fence.wait_workspace_reusable()

    @staticmethod
    def _source_futures(result: Any) -> list[Future[Any]]:
        if result is None:
            return []
        if isinstance(result, Future):
            return [result]
        if isinstance(result, (list, tuple)):
            futures = list(result)
            if not all(isinstance(future, Future) for future in futures):
                raise TypeError("Connector save hook returned a non-Future source handle")
            return futures
        raise TypeError(
            "Connector save hook must return None, Future, or a sequence of Futures"
        )

    def _abort_fence(self, fence: LayerWorkspaceFence) -> None:
        # Close registration before waiting for all readers and compute.
        fence.close_registration()
        fence.wait_workspace_reusable()

    # ==============================
    # Layerwise workspace ordering
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if self._layerwise_workspace_fences_enabled:
            self._wait_for_active_workspace()
            self._execution_id += 1
            self._layer_id = 0
            self._workspace_generation += 1
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._layerwise_workspace_fences_enabled:
            self._wait_for_active_workspace()
        super().wait_for_layer_load(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs
    ) -> Any:
        if not self._layerwise_workspace_fences_enabled:
            return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

        key = WorkspaceKey(
            execution_id=self._execution_id,
            layer_id=self._layer_id,
            arena_id=self._workspace_arena_id,
            workspace_generation=self._workspace_generation,
        )
        fence = LayerWorkspaceFence(key)
        self._active_workspace_fence = fence
        source_futures: list[Future[Any]] = []
        try:
            # This hook runs after the layer attention and output projection
            # have been submitted to the current stream. Store GET is issued
            # by a CPU thread and is not ordered by that stream, so the next
            # layer must not overwrite the shared workspace until this event
            # completes. This is intentionally separate from kv_ready_event,
            # which lets PUT/D2RH overlap with the layer computation.
            compute_release_event = torch.npu.Event()
            compute_release_event.record()
            fence.set_compute_release_event(compute_release_event, key)
            for connector in self._connectors:
                result = connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
                # Register each child result before invoking the next child.
                # If a later connector raises, _abort_fence() must still wait
                # for an earlier Store/P2P source reader using the shared
                # workspace.
                for future in self._source_futures(result):
                    fence.add_source_future(future, key)
                    source_futures.append(future)
            fence.close_registration()
        except BaseException:
            self._abort_fence(fence)
            self._active_workspace_fence = None
            raise
        self._layer_id += 1
        return source_futures

    def wait_for_save(self):
        if not self._layerwise_workspace_fences_enabled:
            return super().wait_for_save()
        fence = self._active_workspace_fence
        first_error: BaseException | None = None
        if fence is not None:
            try:
                fence.wait_workspace_reusable()
            except BaseException as error:
                first_error = error
            self._active_workspace_fence = None
        # Even when P2P source completion fails, drain Store child cleanup so
        # a failed range session revokes open objects and releases leases.
        for connector in self._connectors:
            try:
                connector.wait_for_save()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        offloaded = False
        for c in self._connectors:
            hook = getattr(c, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
