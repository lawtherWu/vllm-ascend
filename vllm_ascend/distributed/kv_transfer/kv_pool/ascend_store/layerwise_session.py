# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import RequestChunkKey


class LayerwiseStoreBackend(Protocol):
    def exists(self, keys: list[str]) -> list[int]: ...

    def batch_put_session_start(
        self, keys: list[str], sizes: list[int], config=None
    ) -> list[int]: ...

    def batch_put_from_multi_buffer_ranges(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
    ) -> list[int]: ...

    def batch_put_session_end(self, keys: list[str]) -> list[int]: ...

    def batch_put_session_revoke(self, keys: list[str]) -> list[int]: ...

    def batch_get_session_start(self, keys: list[str]) -> list[int]: ...

    def batch_get_into_multi_buffer_ranges(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]: ...

    def batch_get_session_end(self, keys: list[str]) -> int: ...


class PutState(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    REVOKED = "revoked"
    END_FAILED = "end_failed"
    CLEANUP_FAILED = "cleanup_failed"
    ERROR = "error"


class StoreCommitError(RuntimeError):
    pass


MOONCAKE_OBJECT_ALREADY_EXISTS = -705
DEFAULT_RANGE_TRANSFER_BATCH_SIZE = 128


@dataclass(frozen=True)
class ExternalWriterWaitConfig:
    """Timing policy for following a put session owned by another client."""

    completion_timeout_s: float = 5.0
    # Mooncake's default put_start_discard_timeout_sec is 30 seconds. Deployments
    # that override that master setting should configure the matching
    # external_writer_stale_timeout_s connector option.
    stale_timeout_s: float = 30.0
    poll_interval_s: float = 0.1

    def __post_init__(self) -> None:
        if self.completion_timeout_s < 0:
            raise ValueError("completion_timeout_s must be non-negative")
        if self.stale_timeout_s < self.completion_timeout_s:
            raise ValueError(
                "stale_timeout_s must be greater than or equal to "
                "completion_timeout_s"
            )
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")


@dataclass(frozen=True)
class PutBinding:
    key: str
    object_size: int

    def __post_init__(self) -> None:
        if not self.key or self.object_size <= 0:
            raise ValueError("PutBinding requires a key and positive object size")


@dataclass
class PutClaim:
    owned_keys: list[str] = field(default_factory=list)
    followed_keys: list[str] = field(default_factory=list)
    external_followed_keys: list[str] = field(default_factory=list)
    failed_keys: list[str] = field(default_factory=list)
    commit_futures: dict[str, Future[list[int]]] = field(default_factory=dict)


@dataclass
class _PutEntry:
    future: Future[list[int]]
    state: PutState = PutState.OPEN
    owner_execution_id: int | None = None


class StorePutRegistry:
    """Coordinates one native put session per canonical key in a Worker."""

    def __init__(
        self,
        backend: LayerwiseStoreBackend,
        replicate_config=None,
        external_writer_wait_config: ExternalWriterWaitConfig | None = None,
    ):
        self.backend = backend
        self.replicate_config = replicate_config
        self.external_writer_wait_config = (
            external_writer_wait_config or ExternalWriterWaitConfig()
        )
        self._entries: dict[str, _PutEntry] = {}
        self._lock = threading.RLock()
        self._external_writer_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="layerwise-external-writer",
        )

    def _set_owned(
        self,
        execution_id: int,
        binding: PutBinding,
        claim: PutClaim,
    ) -> None:
        with self._lock:
            entry = self._entries.get(binding.key)
            if entry is None:
                entry = _PutEntry(
                    future=Future(),
                    state=PutState.OPEN,
                    owner_execution_id=execution_id,
                )
                self._entries[binding.key] = entry
            else:
                entry.state = PutState.OPEN
                entry.owner_execution_id = execution_id
            if binding.key not in claim.owned_keys:
                claim.owned_keys.append(binding.key)
            claim.commit_futures[binding.key] = entry.future

    def _reserve_external(
        self,
        execution_id: int,
        binding: PutBinding,
        claim: PutClaim,
    ) -> None:
        """Reserve a key while its native session is owned by another worker."""

        with self._lock:
            entry = self._entries.get(binding.key)
            if entry is None:
                entry = _PutEntry(
                    future=Future(),
                    state=PutState.OPEN,
                    owner_execution_id=execution_id,
                )
                self._entries[binding.key] = entry
            if binding.key not in claim.followed_keys:
                claim.followed_keys.append(binding.key)
            if binding.key not in claim.external_followed_keys:
                claim.external_followed_keys.append(binding.key)
            claim.commit_futures[binding.key] = entry.future

    def _set_followed(
        self,
        binding: PutBinding,
        claim: PutClaim,
    ) -> None:
        with self._lock:
            entry = self._entries.get(binding.key)
            if entry is None:
                entry = _PutEntry(future=Future())
                self._entries[binding.key] = entry
            entry.state = PutState.COMMITTED
            entry.owner_execution_id = None
            if not entry.future.done():
                entry.future.set_result([0])
            if binding.key not in claim.followed_keys:
                claim.followed_keys.append(binding.key)
            claim.commit_futures[binding.key] = entry.future

    def _set_failed(
        self,
        execution_id: int,
        binding: PutBinding,
        claim: PutClaim,
        error: BaseException,
    ) -> None:
        with self._lock:
            entry = self._entries.get(binding.key)
            if entry is None:
                entry = _PutEntry(future=Future())
                self._entries[binding.key] = entry
            entry.state = PutState.ERROR
            entry.owner_execution_id = execution_id
            if not entry.future.done():
                entry.future.set_exception(error)
            if binding.key not in claim.failed_keys:
                claim.failed_keys.append(binding.key)
            claim.commit_futures[binding.key] = entry.future

    def _completed_keys(self, keys: list[str]) -> set[str]:
        result = self.backend.exists(keys)
        if len(result) != len(keys):
            raise RuntimeError(
                "Mooncake exists returned an invalid result length: "
                f"expected={len(keys)}, actual={len(result)}"
            )
        invalid = {
            key: int(code)
            for key, code in zip(keys, result, strict=True)
            if int(code) not in (0, 1)
        }
        if invalid:
            raise StoreCommitError(
                f"Mooncake exists failed while following external puts: {invalid!r}"
            )
        return {
            key
            for key, code in zip(keys, result, strict=True)
            if int(code) == 1
        }

    def _wait_for_external_commits(
        self,
        bindings: list[PutBinding],
        timeout_s: float,
    ) -> tuple[list[PutBinding], list[PutBinding]]:
        pending = list(bindings)
        completed: list[PutBinding] = []
        deadline = time.monotonic() + timeout_s
        while pending:
            completed_keys = self._completed_keys(
                [binding.key for binding in pending]
            )
            if completed_keys:
                completed.extend(
                    binding
                    for binding in pending
                    if binding.key in completed_keys
                )
                pending = [
                    binding
                    for binding in pending
                    if binding.key not in completed_keys
                ]
            if not pending:
                break
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            time.sleep(
                min(
                    self.external_writer_wait_config.poll_interval_s,
                    remaining_s,
                )
            )
        return completed, pending

    def _resolve_external_writers(
        self,
        execution_id: int,
        bindings: list[PutBinding],
        claim: PutClaim,
    ) -> None:
        config = self.external_writer_wait_config
        completed, pending = self._wait_for_external_commits(
            bindings,
            config.completion_timeout_s,
        )
        for binding in completed:
            self._set_followed(binding, claim)
        if not pending:
            return

        # Mooncake reports an in-progress object as absent from exists(), while
        # PutStart reports OBJECT_ALREADY_EXISTS. Continue watching until the
        # master's stale PutStart window has elapsed, then make one new claim.
        stale_wait_s = config.stale_timeout_s - config.completion_timeout_s
        completed, pending = self._wait_for_external_commits(
            pending,
            stale_wait_s,
        )
        for binding in completed:
            self._set_followed(binding, claim)
        if not pending:
            return

        keys = [binding.key for binding in pending]
        retry_result = self.backend.batch_put_session_start(
            keys,
            [binding.object_size for binding in pending],
            self.replicate_config,
        )
        if len(retry_result) != len(keys):
            raise RuntimeError(
                "Mooncake put_start retry returned an invalid result length"
            )

        still_contended: list[PutBinding] = []
        for binding, code in zip(pending, retry_result, strict=True):
            if code == 0:
                # This claim has already been returned as a follower and its
                # range session will not write this key.  Let the scheduler
                # retry the chunk and become the owner in a fresh execution.
                self._set_failed(
                    execution_id,
                    binding,
                    claim,
                    StoreCommitError(
                        "external writer session expired; key became claimable "
                        f"for the next execution: {binding.key}"
                    ),
                )
            elif code == MOONCAKE_OBJECT_ALREADY_EXISTS:
                still_contended.append(binding)
            else:
                self._set_failed(
                    execution_id,
                    binding,
                    claim,
                    StoreCommitError(
                        f"put_start retry failed for {binding.key}: {code}"
                    ),
                )

        if not still_contended:
            return
        completed, still_contended = self._wait_for_external_commits(
            still_contended,
            config.completion_timeout_s,
        )
        for binding in completed:
            self._set_followed(binding, claim)
        for binding in still_contended:
            self._set_failed(
                execution_id,
                binding,
                claim,
                StoreCommitError(
                    "external writer did not complete before the stale "
                    f"PutStart timeout ({config.stale_timeout_s:.1f}s), and "
                    f"put_start retry failed for {binding.key}: "
                    f"{MOONCAKE_OBJECT_ALREADY_EXISTS}"
                ),
            )

    def claim_execution(
        self,
        execution_id: int,
        ordered_bindings: list[PutBinding],
    ) -> PutClaim:
        """Claim keys in scheduler order and open at most one batch session."""

        unique: list[PutBinding] = []
        externally_owned: list[PutBinding] = []
        claim = PutClaim()
        with self._lock:
            seen: set[str] = set()
            for binding in ordered_bindings:
                if binding.key in seen:
                    claim.followed_keys.append(binding.key)
                    continue
                seen.add(binding.key)
                existing = self._entries.get(binding.key)
                if existing is not None and existing.state in {
                    PutState.ERROR,
                    PutState.REVOKED,
                }:
                    # Failed sessions are terminal for their execution only.
                    # Do not turn a later scheduler retry into a follower of a
                    # stale failed future.
                    self._entries.pop(binding.key, None)
                    existing = None
                if existing is not None and existing.state in {
                    PutState.END_FAILED,
                    PutState.CLEANUP_FAILED,
                }:
                    # Mooncake retains the native put session when put_end or
                    # put_revoke fails. Do not open a second session for the
                    # same canonical key until native cleanup is confirmed.
                    claim.failed_keys.append(binding.key)
                    claim.commit_futures[binding.key] = existing.future
                    continue
                if existing is not None:
                    claim.followed_keys.append(binding.key)
                    claim.commit_futures[binding.key] = existing.future
                    continue
                unique.append(binding)

            if unique:
                keys = [binding.key for binding in unique]
                try:
                    result = self.backend.batch_put_session_start(
                        keys,
                        [binding.object_size for binding in unique],
                        self.replicate_config,
                    )
                    if len(result) != len(keys):
                        raise RuntimeError("Mooncake put_start returned an invalid result length")
                except BaseException as error:
                    for binding in unique:
                        self._set_failed(
                            execution_id,
                            binding,
                            claim,
                            error,
                        )
                else:
                    for binding, code in zip(unique, result, strict=True):
                        if code == 0:
                            self._set_owned(execution_id, binding, claim)
                        elif code == MOONCAKE_OBJECT_ALREADY_EXISTS:
                            externally_owned.append(binding)
                        else:
                            self._set_failed(
                                execution_id,
                                binding,
                                claim,
                                StoreCommitError(
                                    f"put_start failed for {binding.key}: {code}"
                                ),
                            )
                    for binding in externally_owned:
                        self._reserve_external(execution_id, binding, claim)
        # Do not hold the registry lock, or the model-forward thread, while
        # polling another worker's native session. Long prefill sequences can
        # contain many chunks; waiting synchronously here makes every chunk
        # pay the external-writer timeout before it can compute anything.
        if externally_owned:
            def resolve() -> None:
                try:
                    self._resolve_external_writers(
                        execution_id,
                        externally_owned,
                        claim,
                    )
                except BaseException as error:
                    for binding in externally_owned:
                        future = claim.commit_futures.get(binding.key)
                        if future is None or not future.done():
                            self._set_failed(execution_id, binding, claim, error)

            try:
                self._external_writer_executor.submit(resolve)
            except BaseException as error:
                for binding in externally_owned:
                    self._set_failed(execution_id, binding, claim, error)
        with self._lock:
            for binding in ordered_bindings:
                entry = self._entries.get(binding.key)
                if entry is not None:
                    claim.commit_futures.setdefault(binding.key, entry.future)
        return claim

    def publish_commit_result(
        self,
        keys: list[str],
        result_or_error: list[int] | BaseException,
    ) -> None:
        with self._lock:
            if isinstance(result_or_error, BaseException):
                for key in keys:
                    entry = self._entries.get(key)
                    if entry is None or entry.future.done():
                        continue
                    entry.state = PutState.ERROR
                    entry.future.set_exception(result_or_error)
                return
            if len(keys) != len(result_or_error):
                error = RuntimeError("Mooncake put_end returned an invalid result length")
                return self.publish_commit_result(keys, error)
            for key, code in zip(keys, result_or_error, strict=True):
                entry = self._entries.get(key)
                if entry is None or entry.future.done():
                    continue
                if code == 0:
                    entry.state = PutState.COMMITTED
                    entry.future.set_result([code])
                else:
                    entry.state = PutState.END_FAILED
                    entry.future.set_exception(
                        StoreCommitError(f"put_end failed for {key}: {code}")
                    )

    def wait_followed(self, keys: list[str], timeout: float | None = None) -> None:
        if timeout is None:
            config = self.external_writer_wait_config
            timeout = config.stale_timeout_s + config.completion_timeout_s
        deadline = time.monotonic() + timeout
        for key in keys:
            with self._lock:
                entry = self._entries.get(key)
            if entry is None:
                raise StoreCommitError(f"Missing followed put entry: {key}")
            remaining = max(0.0, deadline - time.monotonic())
            try:
                entry.future.result(timeout=remaining)
            except TimeoutError as error:
                terminal_error = StoreCommitError(
                    f"Followed put did not complete within {timeout:.1f}s: {key}"
                )
                with self._lock:
                    current = self._entries.get(key)
                    if current is not None and not current.future.done():
                        current.state = PutState.ERROR
                        current.future.set_exception(terminal_error)
                raise terminal_error from error

    def completed_followed(self, keys: list[str]) -> list[str]:
        """Return followed keys whose external object is already committed."""

        with self._lock:
            return [
                key
                for key in keys
                if (
                    (entry := self._entries.get(key)) is not None
                    and entry.state == PutState.COMMITTED
                    and entry.future.done()
                )
            ]

    def mark_native_state(self, keys: list[str], state: PutState) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.state = state

    def publish_revoke_result(
        self,
        keys: list[str],
        result_or_error: list[int] | BaseException,
        terminal_error: BaseException,
    ) -> None:
        with self._lock:
            if isinstance(result_or_error, BaseException):
                for key in keys:
                    entry = self._entries.get(key)
                    if entry is None:
                        continue
                    entry.state = PutState.CLEANUP_FAILED
                    if not entry.future.done():
                        entry.future.set_exception(terminal_error)
                return
            if len(keys) != len(result_or_error):
                error = RuntimeError("Mooncake put_revoke returned an invalid result length")
                return self.publish_revoke_result(keys, error, terminal_error)
            for key, code in zip(keys, result_or_error, strict=True):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.state = PutState.REVOKED if code == 0 else PutState.CLEANUP_FAILED
                if not entry.future.done():
                    entry.future.set_exception(terminal_error)

    def forget(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if (
                    entry is not None
                    and entry.future.done()
                    and entry.state
                    in {
                        PutState.COMMITTED,
                        PutState.REVOKED,
                        PutState.ERROR,
                    }
                ):
                    self._entries.pop(key, None)


@dataclass
class _ReadEntry:
    references: set[tuple[str, int]] = field(default_factory=set)
    inflight_ranges: int = 0


class StoreReadLeaseRegistry:
    """Reference-counts complete objects shared by requests in one client."""

    def __init__(self, backend: LayerwiseStoreBackend):
        self.backend = backend
        self._entries: dict[str, _ReadEntry] = {}
        self._lock = threading.RLock()

    def _rollback_unreferenced_starts(
        self,
        keys: list[str],
        result: list[int],
        candidates: set[str],
    ) -> None:
        started = [
            key
            for key, code in zip(keys, result, strict=False)
            if code == 0 and key in candidates
        ]
        if not started:
            return
        try:
            code = self.backend.batch_get_session_end(started)
        except BaseException as error:
            raise StoreCommitError(
                f"Failed to roll back Store get sessions: {started!r}"
            ) from error
        if code != 0:
            raise StoreCommitError(
                f"Failed to roll back Store get sessions: {started!r}, {code}"
            )

    def acquire(self, owner: tuple[str, int], keys: list[str]) -> None:
        unique = list(dict.fromkeys(keys))
        to_start: list[str] = []
        with self._lock:
            for key in unique:
                entry = self._entries.setdefault(key, _ReadEntry())
                if not entry.references:
                    to_start.append(key)
            if to_start:
                try:
                    result = self.backend.batch_get_session_start(to_start)
                except BaseException:
                    for key in to_start:
                        if not self._entries[key].references:
                            self._entries.pop(key, None)
                    raise
                if len(result) != len(to_start) or any(
                    code != 0 for code in result
                ):
                    cleanup_error: BaseException | None = None
                    try:
                        self._rollback_unreferenced_starts(
                            to_start, result, set(to_start)
                        )
                    except BaseException as error:
                        cleanup_error = error
                    finally:
                        for key in to_start:
                            if not self._entries[key].references:
                                self._entries.pop(key, None)
                    if cleanup_error is not None:
                        raise StoreCommitError(
                            "batch_get_session_start failed and rollback "
                            f"also failed: {to_start!r}, {result!r}"
                        ) from cleanup_error
                    raise StoreCommitError(
                        "batch_get_session_start failed: "
                        f"{to_start!r}, {result!r}"
                    )
            for key in unique:
                self._entries[key].references.add(owner)

    def refresh(self, owner: tuple[str, int], keys: list[str]) -> None:
        """Refresh the native get sessions for an owner's complete history."""
        unique = list(dict.fromkeys(keys))
        if not unique:
            return
        with self._lock:
            created: list[str] = []
            for key in unique:
                entry = self._entries.get(key)
                if entry is None:
                    entry = _ReadEntry()
                    self._entries[key] = entry
                    created.append(key)
                elif entry.inflight_ranges:
                    raise RuntimeError(
                        f"Cannot refresh Store key {key!r} with an in-flight range"
                    )
            try:
                result = self.backend.batch_get_session_start(unique)
            except BaseException:
                for key in created:
                    if not self._entries[key].references:
                        self._entries.pop(key, None)
                raise
            if len(result) != len(unique) or any(
                code != 0 for code in result
            ):
                cleanup_error: BaseException | None = None
                try:
                    self._rollback_unreferenced_starts(
                        unique, result, set(created)
                    )
                except BaseException as error:
                    cleanup_error = error
                finally:
                    for key in created:
                        if not self._entries[key].references:
                            self._entries.pop(key, None)
                if cleanup_error is not None:
                    raise StoreCommitError(
                        "batch_get_session_start refresh failed and rollback "
                        f"also failed: {unique!r}, {result!r}"
                    ) from cleanup_error
                raise StoreCommitError(
                    f"batch_get_session_start failed: {unique!r}, {result!r}"
                )
            for key in unique:
                self._entries[key].references.add(owner)

    def begin_range(self, owner: tuple[str, int], keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or owner not in entry.references:
                    raise RuntimeError(f"Owner {owner!r} has no Store lease for {key!r}")
                entry.inflight_ranges += 1

    def end_range(self, keys: list[str]) -> None:
        with self._lock:
            for key in keys:
                entry = self._entries[key]
                entry.inflight_ranges -= 1
                if entry.inflight_ranges < 0:
                    raise RuntimeError("Store range completion counted twice")

    def release(self, owner: tuple[str, int], keys: list[str]) -> None:
        to_end: list[str] = []
        with self._lock:
            for key in list(dict.fromkeys(keys)):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.references.discard(owner)
                if not entry.references:
                    if entry.inflight_ranges:
                        raise RuntimeError(
                            f"Cannot release Store key {key!r} with an in-flight range"
                        )
                    to_end.append(key)
            if to_end:
                try:
                    code = self.backend.batch_get_session_end(to_end)
                except BaseException as error:
                    # The native read session is no longer safe to reuse.
                    # Drop local entries even when the backend reports an end
                    # failure, then propagate the failure to the request.
                    for key in to_end:
                        self._entries.pop(key, None)
                    raise StoreCommitError(
                        f"batch_get_session_end failed: {to_end!r}"
                    ) from error
                for key in to_end:
                    self._entries.pop(key, None)
                if code != 0:
                    raise StoreCommitError(f"batch_get_session_end failed: {to_end!r}, {code}")


class RequestStoreLease:
    def __init__(
        self,
        owner: tuple[str, int],
        registry: StoreReadLeaseRegistry,
        range_batch_size: int = DEFAULT_RANGE_TRANSFER_BATCH_SIZE,
    ):
        if range_batch_size <= 0:
            raise ValueError("range_batch_size must be positive")
        self.owner = owner
        self.registry = registry
        self.range_batch_size = range_batch_size
        self.history_keys: set[str] = set()
        self.closed = False

    def ensure_history(self, keys: list[str]) -> None:
        if self.closed:
            raise RuntimeError("Request Store lease is closed")
        missing = sorted(set(keys) - self.history_keys)
        if missing:
            self.registry.acquire(self.owner, missing)
            self.history_keys.update(missing)

    def refresh_history(self, keys: list[str]) -> None:
        if self.closed:
            raise RuntimeError("Request Store lease is closed")
        history = sorted(set(keys))
        if history:
            self.registry.refresh(self.owner, history)
            self.history_keys.update(history)

    def load_layer(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_src_offsets: list[list[int]],
    ) -> list[int]:
        self.registry.begin_range(self.owner, keys)
        try:
            result: list[int] = []
            for start in range(0, len(keys), self.range_batch_size):
                end = start + self.range_batch_size
                result.extend(
                    self.registry.backend.batch_get_into_multi_buffer_ranges(
                        keys[start:end],
                        all_buffer_ptrs[start:end],
                        all_sizes[start:end],
                        all_src_offsets[start:end],
                    )
                )
            return result
        finally:
            self.registry.end_range(keys)

    def release(self) -> None:
        if self.closed:
            return
        try:
            self.registry.release(self.owner, sorted(self.history_keys))
        finally:
            # A failed batch_get_session_end is terminal for this lease; retaining the
            # owner keys would make a retry call the native end operation twice.
            self.history_keys.clear()
            self.closed = True


class ChunkStoreSession:
    """Current-chunk writer; it never closes the request history lease."""

    def __init__(
        self,
        request_chunk: RequestChunkKey,
        backend: LayerwiseStoreBackend,
        request_lease: RequestStoreLease,
        put_registry: StorePutRegistry,
        range_batch_size: int = DEFAULT_RANGE_TRANSFER_BATCH_SIZE,
    ):
        if range_batch_size <= 0:
            raise ValueError("range_batch_size must be positive")
        self.request_chunk = request_chunk
        self.backend = backend
        self.request_lease = request_lease
        self.put_registry = put_registry
        self.range_batch_size = range_batch_size
        self.put_keys: list[str] = []
        self.followed_put_keys: list[str] = []
        self.external_followed_put_keys: list[str] = []
        self.put_state: dict[str, PutState] = {}
        self.put_end_attempted = False
        self.closed = False

    def start(
        self,
        history_keys: list[str],
        claim: PutClaim,
    ) -> None:
        # Bind the native put ownership before acquiring history. If history
        # setup fails, the caller can still revoke every successfully opened
        # put session through revoke_uncommitted().
        self.put_keys = list(dict.fromkeys(claim.owned_keys))
        self.followed_put_keys = list(dict.fromkeys(claim.followed_keys))
        self.external_followed_put_keys = list(
            dict.fromkeys(claim.external_followed_keys)
        )
        self.put_state = {key: PutState.OPEN for key in self.put_keys}
        self.request_lease.ensure_history(history_keys)
        if claim.failed_keys:
            details: dict[str, str] = {}
            for key in claim.failed_keys:
                future = claim.commit_futures.get(key)
                error = future.exception() if future is not None and future.done() else None
                details[key] = repr(error) if error is not None else "unknown error"
            raise StoreCommitError(f"put_start failed: {details!r}")

    def save_layer(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        all_sizes: list[list[int]],
        all_dst_offsets: list[list[int]],
        ready_event=None,
    ) -> list[int]:
        if self.closed or self.put_end_attempted:
            raise RuntimeError("Cannot save a closed Store chunk")
        if ready_event is not None:
            # This method runs in a background CPU thread and invokes the
            # Mooncake range API directly with raw buffer addresses. A
            # stream-local Event.wait() would not order that CPU API call
            # after the producer stream. Synchronize the event here so the
            # source KV is complete before Mooncake starts reading it.
            ready_event.synchronize()
        result: list[int] = []
        for start in range(0, len(keys), self.range_batch_size):
            end = start + self.range_batch_size
            result.extend(
                self.backend.batch_put_from_multi_buffer_ranges(
                    keys[start:end],
                    all_buffer_ptrs[start:end],
                    all_sizes[start:end],
                    all_dst_offsets[start:end],
                )
            )
        expected_bytes = tuple(sum(row) for row in all_sizes)
        if len(result) != len(keys) or any(
            int(code) != expected
            for code, expected in zip(result, expected_bytes, strict=False)
        ):
            raise StoreCommitError(
                f"Store range put failed: result={result!r}, "
                f"expected={expected_bytes!r}"
            )
        return result

    def commit_owned(self) -> list[int]:
        if self.put_end_attempted:
            raise RuntimeError("Store put_end was already attempted")
        self.put_end_attempted = True
        if not self.put_keys:
            self.closed = True
            return []
        try:
            result = self.backend.batch_put_session_end(self.put_keys)
        except BaseException as error:
            self.put_registry.publish_commit_result(self.put_keys, error)
            for key in self.put_keys:
                self.put_state[key] = PutState.END_FAILED
            self.closed = True
            try:
                self._reconcile_unknown_end(error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"Store put_end failed with {error!r}; cleanup failed with "
                    f"{cleanup_error!r}"
                ) from error
            raise
        if len(result) != len(self.put_keys):
            error = StoreCommitError(
                "Mooncake put_end returned an invalid result length: "
                f"expected={len(self.put_keys)}, actual={len(result)}"
            )
            self.put_registry.publish_commit_result(self.put_keys, error)
            for key in self.put_keys:
                self.put_state[key] = PutState.END_FAILED
            self.closed = True
            try:
                self._reconcile_unknown_end(error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"{error}; cleanup failed with {cleanup_error!r}"
                ) from error
            raise error
        self.put_registry.publish_commit_result(self.put_keys, result)
        for key, code in zip(self.put_keys, result, strict=True):
            self.put_state[key] = PutState.COMMITTED if code == 0 else PutState.END_FAILED
        self.closed = True
        failed_keys = [
            key
            for key, code in zip(self.put_keys, result, strict=True)
            if code != 0
        ]
        if failed_keys:
            error = StoreCommitError(f"Store put_end failed: {result!r}")
            try:
                self._revoke_keys(failed_keys, error)
            except BaseException as cleanup_error:
                raise StoreCommitError(
                    f"{error}; cleanup failed with {cleanup_error!r}"
                ) from error
            raise error
        return result

    def _reconcile_unknown_end(self, terminal_error: BaseException) -> None:
        """Resolve native sessions after an ambiguous put_end result."""

        try:
            exists = self.backend.exists(self.put_keys)
        except BaseException as error:
            self.put_registry.mark_native_state(
                self.put_keys, PutState.CLEANUP_FAILED
            )
            for key in self.put_keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                f"Store complete lookup failed after put_end: {error!r}"
            ) from error
        if len(exists) != len(self.put_keys) or any(code not in (0, 1) for code in exists):
            self.put_registry.mark_native_state(
                self.put_keys, PutState.CLEANUP_FAILED
            )
            for key in self.put_keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                "Store complete lookup returned an invalid result after "
                f"put_end: {exists!r}"
            )

        complete_keys = [
            key
            for key, is_complete in zip(self.put_keys, exists, strict=True)
            if is_complete == 1
        ]
        pending_keys = [
            key
            for key, is_complete in zip(self.put_keys, exists, strict=True)
            if is_complete == 0
        ]
        self.put_registry.mark_native_state(complete_keys, PutState.COMMITTED)
        for key in complete_keys:
            self.put_state[key] = PutState.COMMITTED
        self._revoke_keys(pending_keys, terminal_error)

    def _revoke_keys(
        self,
        keys: list[str],
        terminal_error: BaseException,
    ) -> None:
        if not keys:
            return
        try:
            result = self.backend.batch_put_session_revoke(keys)
        except BaseException as error:
            self.put_registry.publish_revoke_result(keys, error, terminal_error)
            for key in keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise
        self.put_registry.publish_revoke_result(keys, result, terminal_error)
        if len(result) != len(keys):
            for key in keys:
                self.put_state[key] = PutState.CLEANUP_FAILED
            raise StoreCommitError(
                "Mooncake put_revoke returned an invalid result length: "
                f"expected={len(keys)}, actual={len(result)}"
            )
        failed: dict[str, int] = {}
        for key, code in zip(keys, result, strict=True):
            if code == 0:
                self.put_state[key] = PutState.REVOKED
            else:
                self.put_state[key] = PutState.CLEANUP_FAILED
                failed[key] = int(code)
        if failed:
            raise StoreCommitError(f"Store put_revoke failed: {failed!r}")

    def finalize_followed(self, has_more_prefill_chunks: bool) -> None:
        # A follower (whether local or cross-worker) owns no range data. Do
        # not hold a duplicate request open until a potentially multi-GB long
        # chunk has finished uploading elsewhere. Only committed objects are
        # eligible for lease refresh; pending ones can be discovered by the
        # next chunk through exists()/the registry. Owners still wait for all
        # range futures before put_end so their NPU cache buffers are not reused
        # while Mooncake is reading them.
        local_followed = [
            key
            for key in self.followed_put_keys
            if key not in self.external_followed_put_keys
        ]
        completed_external = self.put_registry.completed_followed(
            self.external_followed_put_keys
        )
        completed_local = self.put_registry.completed_followed(local_followed)
        all_keys = list(
            dict.fromkeys(self.put_keys + completed_local + completed_external)
        )
        if has_more_prefill_chunks:
            self.request_lease.refresh_history(
                list(self.request_lease.history_keys) + all_keys
            )

    def revoke_uncommitted(self) -> None:
        if not self.put_keys:
            return
        keys = [
            key
            for key in self.put_keys
            if self.put_state.get(key)
            in {
                PutState.OPEN,
                PutState.END_FAILED,
                PutState.CLEANUP_FAILED,
            }
        ]
        if not keys:
            return
        self._revoke_keys(
            keys,
            StoreCommitError("Store put was revoked before commit"),
        )
