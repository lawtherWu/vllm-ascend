# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""Install the Prefill chunk-alignment hook used by layerwise KV offload.

The vLLM 0.23 scheduler keeps both the running-prefill and first-prefill
allocation paths in ``Scheduler.schedule``.  A normal wrapper cannot change
``num_new_tokens`` after it has been calculated, so this module makes a small,
version-checked source adaptation of that method.  The adapted method calls a
no-op hook on the scheduler object; only the Ascend producer scheduler
overrides the hook.  This keeps the upstream scheduler algorithm in one place
and leaves decode schedulers untouched.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.utils import vllm_version_is


_PATCH_MARKER = "_vllm_ascend_layerwise_offload_patched"
# This is the source fingerprint of Scheduler.schedule in the pinned vLLM
# 0.23.0 checkout.  A source change can move allocation points and must not be
# silently adapted by this patch.
_EXPECTED_SOURCE_SHA256 = "62f610d9a69ac180d3af339fbc4278b6289d65000639c44cadff79e62147155d"
_EXPECTED_SIGNATURE = "(self, throttle_prefills: bool = False) -> SchedulerOutput"


def _default_adjust_num_new_tokens_before_allocate(
    self: Scheduler,
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
    token_budget: int,
) -> int:
    """Preserve upstream scheduling for schedulers that do not opt in."""

    del request, effective_computed_tokens, token_budget
    return proposed_tokens


def align_prefill_chunk_for_store(
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
    token_budget: int,
    granularity: int,
) -> int:
    """Align a non-final prompt chunk to Store's full-block granularity.

    Prefix hits are already included in ``effective_computed_tokens`` by the
    scheduler.  Decode tokens and the final prompt chunk are deliberately
    untouched.  Returning zero asks the caller to retain its original
    RUNNING/WAITING skip semantics when the remaining budget cannot form a
    complete block; the next scheduling step gets a fresh token budget.
    """

    if proposed_tokens <= 0:
        return proposed_tokens
    if token_budget < proposed_tokens:
        raise ValueError("scheduler proposed more tokens than its token budget")
    if granularity <= 0:
        raise ValueError(f"invalid Store granularity: {granularity}")

    prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
    if effective_computed_tokens >= prompt_tokens:
        return proposed_tokens

    prompt_part = min(proposed_tokens, prompt_tokens - effective_computed_tokens)
    # A request may have speculative/decode tokens after its prompt.  If the
    # prompt portion reaches the end, this is the final prefill chunk and the
    # original proposal (including any extra tokens) must be retained.
    if effective_computed_tokens + prompt_part >= prompt_tokens:
        return proposed_tokens

    aligned_end = (
        (effective_computed_tokens + prompt_part) // granularity
    ) * granularity
    aligned_tokens = aligned_end - effective_computed_tokens
    return max(aligned_tokens, 0)


def _make_patched_schedule() -> Callable[..., SchedulerOutput]:
    """Create the adapted vLLM 0.23.0 method from its original source."""

    source = inspect.getsource(Scheduler.schedule)
    normalized = textwrap.dedent(source)
    running_anchor = (
        "        if num_new_tokens == 0:\n"
        "            # The request cannot be scheduled because one of the following\n"
    )
    waiting_anchor = "            # Skip block alignment when setting up async receive (no local work).\n"
    if normalized.count(running_anchor) != 1:
        raise RuntimeError("unexpected Scheduler.schedule RUNNING allocation marker")
    if normalized.count(waiting_anchor) != 1:
        raise RuntimeError("unexpected Scheduler.schedule WAITING allocation marker")

    running_hook = (
        "            num_new_tokens = self._adjust_num_new_tokens_before_allocate(\n"
        "                request=request,\n"
        "                effective_computed_tokens=request.num_computed_tokens,\n"
        "                proposed_tokens=num_new_tokens,\n"
        "                token_budget=token_budget,\n"
        "            )\n\n"
    )
    normalized = normalized.replace(running_anchor, running_hook + running_anchor, 1)

    waiting_hook = (
        "            if not load_kv_async:\n"
        "                num_new_tokens = self._adjust_num_new_tokens_before_allocate(\n"
        "                    request=request,\n"
        "                    effective_computed_tokens=num_computed_tokens,\n"
        "                    proposed_tokens=num_new_tokens,\n"
        "                    token_budget=token_budget,\n"
        "                )\n"
        "                if num_new_tokens == 0:\n"
        "                    break\n\n"
    )
    normalized = normalized.replace(waiting_anchor, waiting_hook + waiting_anchor, 1)

    namespace = dict(vars(inspect.getmodule(Scheduler)))
    exec(compile(normalized, inspect.getsourcefile(Scheduler) or "vllm_scheduler.py", "exec"), namespace)
    patched = namespace.get("schedule")
    if not callable(patched):
        raise RuntimeError("failed to build patched Scheduler.schedule")
    return patched


def apply_layerwise_offload_scheduler_patch() -> None:
    """Patch the original Scheduler class exactly once at import time."""

    current = Scheduler.schedule
    if getattr(current, _PATCH_MARKER, False):
        return
    if not vllm_version_is("0.23.0"):
        raise RuntimeError(
            "Layerwise Host KV offload requires vLLM 0.23.0; refusing to patch "
            f"Scheduler.schedule for {getattr(__import__('vllm'), '__version__', 'unknown')}"
        )
    if str(inspect.signature(current)) != _EXPECTED_SIGNATURE:
        raise RuntimeError(
            "Unsupported vLLM Scheduler.schedule signature: "
            f"{inspect.signature(current)}"
        )
    source = inspect.getsource(current)
    if hashlib.sha256(source.encode()).hexdigest() != _EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Unsupported vLLM Scheduler.schedule source; expected the pinned "
            "vLLM 0.23.0 allocation layout"
        )
    if "Schedule newly needed KV blocks" not in source:
        raise RuntimeError("Scheduler.schedule allocation marker is missing")

    Scheduler._adjust_num_new_tokens_before_allocate = (  # type: ignore[attr-defined]
        _default_adjust_num_new_tokens_before_allocate
    )
    patched = _make_patched_schedule()
    setattr(patched, _PATCH_MARKER, True)
    setattr(patched, "_vllm_ascend_original", current)
    Scheduler.schedule = patched  # type: ignore[method-assign]


class LayerwiseOffloadAsyncScheduler(AsyncScheduler):
    """AsyncScheduler variant that aligns Prefill chunks for Store writes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.store_granularity = int(getattr(self.cache_config, "block_size", self.block_size))

    def _adjust_num_new_tokens_before_allocate(
        self,
        request: Any,
        effective_computed_tokens: int,
        proposed_tokens: int,
        token_budget: int,
    ) -> int:
        # The hook is reached on decode scheduling too.  Only requests whose
        # current scheduler state is a prompt chunk are eligible.
        if effective_computed_tokens >= int(getattr(request, "num_prompt_tokens", 0)):
            return proposed_tokens
        return align_prefill_chunk_for_store(
            request,
            effective_computed_tokens,
            proposed_tokens,
            token_budget,
            self.store_granularity,
        )
