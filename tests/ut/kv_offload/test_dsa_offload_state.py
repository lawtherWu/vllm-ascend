# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.dsa_offload_state import (
    DsaLayerWorkspace,
    DsaResidentState,
    ResidentOwner,
)


def _workspace(layer_id=0):
    return DsaLayerWorkspace(
        layer_id=layer_id,
        resident_kv_cache=torch.empty((3, 128, 1, 4)),
        resident_k_rope=torch.empty((3, 128, 1, 2)),
        selection_kv_cache=torch.empty((2, 2048, 1, 4)),
        selection_k_rope=torch.empty((2, 2048, 1, 2)),
        selection_block_table=torch.full((2, 16), -1, dtype=torch.int32),
    )


def test_resident_rows_are_invalid_until_bound_and_begin_step_applies_pending():
    state = DsaResidentState([_workspace()])
    owner = ResidentOwner("req-a", 0)
    assert not state.row_state(0, 1).valid
    state.bind_row(0, 1, owner)
    assert state.row_state(0, 1).owner == owner
    state.invalidate_rows({0: [1]})
    # Invalidation is deferred until the previous install fence is safe and
    # the next step starts.
    assert state.row_state(0, 1).valid
    state.begin_step()
    assert not state.row_state(0, 1).valid
    assert state.rows_for_owner(owner, 0) == ()


def test_resident_state_waits_for_previous_install_event():
    class Event:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    event = Event()
    state = DsaResidentState([_workspace()])
    state.record_final_install_event(event)
    state.begin_step()
    assert event.waited


def test_resident_state_waits_for_every_group_event_once():
    class Event:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    events = (Event(), Event())
    state = DsaResidentState([_workspace()])
    for event in events:
        state.record_final_install_event(event)

    state.begin_step()
    assert [event.wait_count for event in events] == [1, 1]
    # Consumed events must not be retained and waited again on a later step.
    state.begin_step()
    assert [event.wait_count for event in events] == [1, 1]
