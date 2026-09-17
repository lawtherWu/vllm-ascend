# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future

import pytest

from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import (
    LayerWorkspaceFence,
    RequestChunkKey,
    WorkspaceIdentityError,
    WorkspaceKey,
    WorkspaceTransferError,
)


def test_request_and_workspace_identity_validation():
    assert RequestChunkKey("request", 1, 2).chunk_id == 2
    with pytest.raises(ValueError):
        RequestChunkKey("", 0, 0)
    with pytest.raises(ValueError):
        WorkspaceKey(0, -1, 0, 0)


def test_fence_waits_for_every_registered_source():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    first: Future[None] = Future()
    second: Future[None] = Future()
    fence.add_source_future(first, key)
    fence.add_source_future(second, key)
    fence.close_registration()
    first.set_result(None)
    assert fence.pending == 1
    with pytest.raises(TimeoutError):
        fence.wait_source_safe(timeout=0)
    second.set_result(None)
    fence.wait_source_safe(timeout=0)


def test_fence_rejects_wrong_generation_before_counting():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    with pytest.raises(WorkspaceIdentityError):
        fence.add_source_future(Future(), WorkspaceKey(1, 2, 3, 5))
    assert fence.pending == 0


def test_fence_propagates_failure_and_cancel():
    key = WorkspaceKey(1, 2, 3, 4)
    for complete in (
        lambda future: future.set_exception(ValueError("transfer failed")),
        lambda future: future.cancel(),
    ):
        fence = LayerWorkspaceFence(key)
        future: Future[None] = Future()
        fence.add_source_future(future, key)
        fence.close_registration()
        complete(future)
        with pytest.raises(WorkspaceTransferError):
            fence.wait_source_safe(timeout=0)


def test_empty_fence_completes_after_registration_closes():
    fence = LayerWorkspaceFence(WorkspaceKey(1, 2, 3, 4))
    fence.close_registration()
    fence.wait_source_safe(timeout=0)
