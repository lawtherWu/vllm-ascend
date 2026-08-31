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

import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput

_PATCH_MARKER = "_vllm_ascend_layerwise_offload_patched"


def _default_adjust_num_new_tokens_before_allocate(
    self: Scheduler,
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
) -> int:
    """Leave non-layerwise schedulers unchanged."""

    del self, request, effective_computed_tokens
    return proposed_tokens


def align_prefill_chunk_for_store(
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
    granularity: int,
) -> int:
    """Align non-final prompt chunks to Store's block granularity."""

    if proposed_tokens <= 0:
        return proposed_tokens
    if granularity <= 0:
        raise ValueError(f"invalid Store granularity: {granularity}")

    prompt_tokens = request.num_prompt_tokens
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
        "            )\n\n"
    )
    normalized = normalized.replace(running_anchor, running_hook + running_anchor, 1)

    waiting_hook = (
        "            if not load_kv_async:\n"
        "                num_new_tokens = self._adjust_num_new_tokens_before_allocate(\n"
        "                    request=request,\n"
        "                    effective_computed_tokens=num_computed_tokens,\n"
        "                    proposed_tokens=num_new_tokens,\n"
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
    source = inspect.getsource(current)

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
        self.store_granularity = int(self.cache_config.block_size)

    def _adjust_num_new_tokens_before_allocate(
        self,
        request: Any,
        effective_computed_tokens: int,
        proposed_tokens: int,
    ) -> int:
        return align_prefill_chunk_for_store(
            request,
            effective_computed_tokens,
            proposed_tokens,
            self.store_granularity,
        )
