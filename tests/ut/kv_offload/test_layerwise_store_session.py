# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_session import (
    ChunkStoreSession,
    PutBinding,
    RequestStoreLease,
    StoreCommitError,
    StorePutRegistry,
    StoreReadLeaseRegistry,
)
from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import RequestChunkKey


class FakeBackend:
    def __init__(self):
        self.calls = Counter()
        self.get_start_keys = []
        self.put_start_result = None
        self.put_end_result = None
        self.put_range_result = None

    def batch_put_session_start(self, keys, sizes, config=None):
        del sizes, config
        self.calls["put_start"] += 1
        return self.put_start_result or [0] * len(keys)

    def batch_put_from_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        del ptrs, offsets
        self.calls["put_range"] += 1
        return self.put_range_result or [sum(row) for row in sizes]

    def batch_put_session_end(self, keys):
        self.calls["put_end"] += 1
        return self.put_end_result or [0] * len(keys)

    def batch_put_session_revoke(self, keys):
        self.calls["put_revoke"] += 1
        return [0] * len(keys)

    def batch_get_session_start(self, keys):
        self.calls["get_start"] += 1
        self.get_start_keys.append(list(keys))
        return [0] * len(keys)

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        del ptrs, offsets
        self.calls["get_range"] += 1
        return [sum(row) for row in sizes]

    def batch_get_session_end(self, keys):
        del keys
        self.calls["get_end"] += 1
        return 0


def test_same_execution_key_has_one_writer_and_follower():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(
        1,
        [PutBinding("same", 128), PutBinding("same", 128)],
    )
    assert backend.calls["put_start"] == 1
    assert claim.owned_keys == ["same"]
    assert claim.followed_keys == ["same"]
    registry.publish_commit_result(["same"], [0])
    registry.wait_followed(claim.followed_keys, timeout=0)


def test_owner_failure_resolves_follower_future():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(
        1,
        [PutBinding("same", 128), PutBinding("same", 128)],
    )
    registry.publish_commit_result(["same"], RuntimeError("commit failed"))
    with pytest.raises(RuntimeError, match="commit failed"):
        registry.wait_followed(claim.followed_keys, timeout=0)


def test_put_start_failure_reports_backend_code():
    backend = FakeBackend()
    backend.put_start_result = [-2]
    registry = StorePutRegistry(backend)
    claim = registry.claim_execution(1, [PutBinding("duplicate", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0),
        backend,
        RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend)),
        registry,
    )

    with pytest.raises(StoreCommitError, match=r"duplicate.*-2"):
        session.start([], claim)


def test_request_lease_is_reference_counted_across_requests():
    backend = FakeBackend()
    registry = StoreReadLeaseRegistry(backend)
    first = RequestStoreLease(("a", 0), registry)
    second = RequestStoreLease(("b", 0), registry)
    first.ensure_history(["prefix"])
    second.ensure_history(["prefix"])
    assert backend.calls["get_start"] == 1
    first.release()
    assert backend.calls["get_end"] == 0
    second.release()
    assert backend.calls["get_end"] == 1


def test_chunk_commit_then_retains_new_history_for_next_chunk():
    backend = FakeBackend()
    put_registry = StorePutRegistry(backend)
    read_registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), read_registry)
    claim = put_registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0),
        backend,
        lease,
        put_registry,
    )
    session.start([], claim)
    session.save_layer(["new"], [[1000]], [[64]], [[0]])
    session.commit_owned()
    session.finalize_followed(has_more_prefill_chunks=True)
    assert lease.history_keys == {"new"}
    assert backend.calls["put_end"] == 1
    assert backend.calls["get_start"] == 1


def test_chunk_commit_refreshes_complete_request_history():
    backend = FakeBackend()
    put_registry = StorePutRegistry(backend)
    read_registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), read_registry)
    lease.ensure_history(["old-0", "old-1"])
    claim = put_registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 1),
        backend,
        lease,
        put_registry,
    )
    session.start(sorted(lease.history_keys), claim)
    session.save_layer(["new"], [[1000]], [[64]], [[0]])
    session.commit_owned()
    session.finalize_followed(has_more_prefill_chunks=True)

    assert lease.history_keys == {"old-0", "old-1", "new"}
    assert backend.get_start_keys == [
        ["old-0", "old-1"],
        ["new", "old-0", "old-1"],
    ]


def test_request_lease_rejects_refresh_with_inflight_range():
    backend = FakeBackend()
    registry = StoreReadLeaseRegistry(backend)
    lease = RequestStoreLease(("request", 0), registry)
    lease.ensure_history(["history"])
    registry.begin_range(lease.owner, ["history"])

    with pytest.raises(RuntimeError, match="in-flight range"):
        lease.refresh_history(["history"])

    registry.end_range(["history"])


def test_range_put_synchronizes_ready_event_before_reading_source():
    backend = FakeBackend()
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)

    call_order = []
    ready_event = MagicMock()
    ready_event.synchronize.side_effect = lambda: call_order.append("synchronize")
    original_put_range = backend.batch_put_from_multi_buffer_ranges

    def traced_put_range(keys, ptrs, sizes, offsets):
        call_order.append("put_range")
        return original_put_range(keys, ptrs, sizes, offsets)

    backend.batch_put_from_multi_buffer_ranges = traced_put_range
    session.save_layer(
        ["new"],
        [[1000]],
        [[64]],
        [[0]],
        ready_event=ready_event,
    )

    assert call_order == ["synchronize", "put_range"]
    ready_event.synchronize.assert_called_once_with()
    ready_event.wait.assert_not_called()


def test_range_put_rejects_unexpected_positive_byte_count():
    backend = FakeBackend()
    backend.put_range_result = [1]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)
    with pytest.raises(StoreCommitError, match="expected"):
        session.save_layer(["new"], [[1000]], [[64]], [[0]])


def test_put_end_failure_is_terminal_and_not_revoked():
    backend = FakeBackend()
    backend.put_end_result = [-7]
    registry = StorePutRegistry(backend)
    lease = RequestStoreLease(("request", 0), StoreReadLeaseRegistry(backend))
    claim = registry.claim_execution(1, [PutBinding("new", 128)])
    session = ChunkStoreSession(
        RequestChunkKey("request", 0, 0), backend, lease, registry
    )
    session.start([], claim)
    with pytest.raises(StoreCommitError):
        session.commit_owned()
    session.revoke_uncommitted()
    assert backend.calls["put_revoke"] == 0
