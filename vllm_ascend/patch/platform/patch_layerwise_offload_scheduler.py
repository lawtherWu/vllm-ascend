# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""Install the Prefill chunk-alignment hook used by layerwise KV offload.

vLLM exposes ``_mamba_block_aligned_split`` immediately before KV block
allocation on both the running-prefill and first-prefill paths.  This module
wraps that narrow extension point instead of copying ``Scheduler.schedule``.
Consequently, changes to the vLLM version, schedule signature, or schedule
implementation do not require this patch to be regenerated.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


_HOOK_NAME = "_mamba_block_aligned_split"
_PATCH_MARKER = "_vllm_ascend_layerwise_offload_patched"
_LAYERWISE_SCHEDULER_MARKER = "_vllm_ascend_layerwise_offload_enabled"
_COMPUTED_TOKEN_DELTA_NAMES = (
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)


def align_prefill_chunk_for_store(
    request: Any,
    effective_computed_tokens: int,
    proposed_tokens: int,
    token_budget: int,
    granularity: int,
) -> int:
    """Align a non-final prompt chunk to Store's full-block granularity.

    Prefix hits are already included in ``effective_computed_tokens`` by the
    scheduler. Decode tokens and the final prompt chunk are deliberately
    untouched. Returning zero preserves the upstream hook's existing RUNNING
    skip and WAITING break semantics.
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
    # A request may have speculative/decode tokens after its prompt. If the
    # prompt portion reaches the end, this is the final prefill chunk and the
    # original proposal (including any extra tokens) must be retained.
    if effective_computed_tokens + prompt_part >= prompt_tokens:
        return proposed_tokens

    aligned_end = ((effective_computed_tokens + prompt_part) // granularity) * granularity
    aligned_tokens = aligned_end - effective_computed_tokens
    return max(aligned_tokens, 0)


def _effective_computed_tokens(
    request: Any,
    computed_token_deltas: tuple[Any, ...],
    hook_kwargs: dict[str, Any],
) -> int:
    """Read the hook's optional computed-token deltas across vLLM revisions."""

    deltas = list(computed_token_deltas[: len(_COMPUTED_TOKEN_DELTA_NAMES)])
    for delta_name in _COMPUTED_TOKEN_DELTA_NAMES[len(deltas) :]:
        deltas.append(hook_kwargs.get(delta_name, 0))
    return int(getattr(request, "num_computed_tokens", 0)) + sum(map(int, deltas))


def _make_block_alignment_hook(original_hook: Callable[..., int]) -> Callable[..., int]:
    """Wrap the upstream hook while preserving it for all other schedulers."""

    @functools.wraps(original_hook)
    def layerwise_block_alignment_hook(
        self: Any,
        request: Any,
        num_new_tokens: int,
        *computed_token_deltas: Any,
        **hook_kwargs: Any,
    ) -> int:
        if not getattr(self, _LAYERWISE_SCHEDULER_MARKER, False):
            return original_hook(
                self,
                request,
                num_new_tokens,
                *computed_token_deltas,
                **hook_kwargs,
            )

        effective_computed_tokens = _effective_computed_tokens(
            request,
            computed_token_deltas,
            hook_kwargs,
        )
        return align_prefill_chunk_for_store(
            request=request,
            effective_computed_tokens=effective_computed_tokens,
            proposed_tokens=num_new_tokens,
            # The upstream scheduler already bounded num_new_tokens by its
            # current budget before invoking this hook.
            token_budget=num_new_tokens,
            granularity=int(self.store_granularity),
        )

    setattr(layerwise_block_alignment_hook, _PATCH_MARKER, True)
    setattr(layerwise_block_alignment_hook, "_vllm_ascend_original", original_hook)
    return layerwise_block_alignment_hook


def apply_layerwise_offload_scheduler_patch() -> None:
    """Patch the upstream pre-allocation hook exactly once.

    This is intentionally a capability check rather than a vLLM version or
    source-layout check. Additive changes to the hook signature are accepted
    by the wrapper's ``*args``/``**kwargs`` forwarding.
    """

    current_hook = getattr(AsyncScheduler, _HOOK_NAME, None)
    if not callable(current_hook):
        raise RuntimeError(
            "Layerwise Host KV offload requires vLLM AsyncScheduler to expose "
            f"a callable {_HOOK_NAME} pre-allocation hook"
        )
    if getattr(current_hook, _PATCH_MARKER, False):
        return

    setattr(AsyncScheduler, _HOOK_NAME, _make_block_alignment_hook(current_hook))


class LayerwiseOffloadAsyncScheduler(AsyncScheduler):
    """AsyncScheduler variant that aligns Prefill chunks for Store writes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, _LAYERWISE_SCHEDULER_MARKER, True)
        self.need_mamba_block_aligned_split = True
        self.store_granularity = int(self.cache_config.block_size)
