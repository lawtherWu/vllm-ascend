# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops import dsa_offload


def test_dsa_config_restricts_first_release_buckets():
    assert dsa_offload.DsaOffloadConfig(raw_seq=4).raw_seq == 4
    with pytest.raises(ValueError):
        dsa_offload.DsaOffloadConfig(raw_seq=2)
    with pytest.raises(ValueError):
        dsa_offload.DsaOffloadConfig(topk=1024)


def test_pa_block_table_validation_accepts_sentinel_and_rejects_bad_rows():
    cache = torch.empty((4, 128, 8), dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, -1], [2, 3, -1]], dtype=torch.int32)
    seq = torch.tensor([256, 256], dtype=torch.int32)
    dsa_offload.validate_full_kv_block_table(table, full_kv_cache=cache, full_kv_actual_seq=seq)

    with pytest.raises(ValueError, match="outside"):
        dsa_offload.validate_full_kv_block_table(
            torch.tensor([[4]], dtype=torch.int32), full_kv_cache=cache
        )
    with pytest.raises(ValueError, match="sentinel"):
        dsa_offload.validate_full_kv_block_table(
            torch.tensor([[-2]], dtype=torch.int32), full_kv_cache=cache
        )


def test_dsa_plan_rejects_shapes_outside_recipes_schema():
    seq = torch.tensor([128], dtype=torch.int32)
    config = dsa_offload.DsaOffloadConfig(raw_seq=1)
    with pytest.raises(ValueError, match="recipes DsaPlan layout"):
        dsa_offload.validate_selection_inputs(
            torch.zeros((1, 2048), dtype=torch.int32), seq, config=config
        )
    with pytest.raises(TypeError, match="torch.int32"):
        dsa_offload.validate_selection_inputs(
            torch.zeros((1, 1, 2048), dtype=torch.int64), seq, config=config
        )

    mtp_config = dsa_offload.DsaOffloadConfig(raw_seq=4)
    dsa_offload.validate_selection_inputs(
        torch.zeros((1, 4, 1, 2048), dtype=torch.int32),
        seq,
        config=mtp_config,
    )
    with pytest.raises(ValueError, match="recipes DsaPlan layout"):
        dsa_offload.validate_selection_inputs(
            torch.zeros((4, 1, 2048), dtype=torch.int32),
            seq,
            config=mtp_config,
        )


def test_dsa_plan_uses_installed_custom_ops(monkeypatch):
    calls = []

    def fake_plan(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.tensor(1), torch.tensor(2), torch.tensor(3)

    monkeypatch.setattr(dsa_offload, "_custom_op", lambda name: fake_plan)
    topk = torch.zeros((1, 1, 2048), dtype=torch.int32)
    result = dsa_offload.dsa_plan(
        topk,
        torch.tensor([128], dtype=torch.int32),
        torch.zeros((1, 16), dtype=torch.int32),
        torch.zeros((1, 128), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    assert len(calls) == 1
    assert result[0].item() == 1


def test_dsa_serve_propagates_an_incompatible_schema_error(monkeypatch):
    def old_schema(*args, **kwargs):
        del args, kwargs
        raise TypeError("old schema")

    monkeypatch.setattr(dsa_offload, "_custom_op", lambda name: old_schema)
    plan = torch.empty((2048, 2), dtype=torch.int32)
    full_cache = torch.empty((2, 128, 1, 8), dtype=torch.bfloat16)
    full_rope = torch.empty((2, 128, 1, 2), dtype=torch.bfloat16)
    pool_cache = torch.empty((1, 16, 8), dtype=torch.bfloat16)
    pool_rope = torch.empty((1, 16, 2), dtype=torch.bfloat16)
    selection_cache = torch.empty((16, 128, 8), dtype=torch.bfloat16)
    selection_rope = torch.empty((16, 128, 2), dtype=torch.bfloat16)
    args = (
        plan,
        full_cache,
        full_rope,
        pool_cache,
        pool_rope,
        selection_cache,
        selection_rope,
    )
    with pytest.raises(TypeError, match="old schema"):
        dsa_offload.dsa_serve(
            *args,
            full_kv_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        )


def test_dsa_serve_normalizes_main_cache_and_uses_pa_schema(monkeypatch):
    calls = []

    def pa_schema(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(dsa_offload, "_custom_op", lambda name: pa_schema)
    plan = torch.empty((2048, 2), dtype=torch.int32)
    # vLLM's MLA cache has a singleton KV-head dimension.
    full_cache = torch.empty((2, 128, 1, 8), dtype=torch.bfloat16)
    full_rope = torch.empty((2, 128, 1, 2), dtype=torch.bfloat16)
    pool_cache = torch.empty((1, 16, 8), dtype=torch.bfloat16)
    pool_rope = torch.empty((1, 16, 2), dtype=torch.bfloat16)
    selection_cache = torch.empty((16, 128, 8), dtype=torch.bfloat16)
    selection_rope = torch.empty((16, 128, 2), dtype=torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    dsa_offload.dsa_serve(
        plan,
        full_cache,
        full_rope,
        pool_cache,
        pool_rope,
        selection_cache,
        selection_rope,
        full_kv_block_table=block_table,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 7
    assert args[1].shape == torch.Size([2, 128, 8])
    assert args[2].shape == torch.Size([2, 128, 2])
    assert kwargs["full_kv_block_table"] is block_table
