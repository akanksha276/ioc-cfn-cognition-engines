# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from distill.app.services import distillation_lock


@pytest.mark.asyncio
async def test_try_begin_run_blocks_second_concurrent_key():
    assert await distillation_lock.try_begin_run("ws", "mas") is True
    assert await distillation_lock.try_begin_run("ws", "mas") is False
    await distillation_lock.end_run("ws", "mas")
    assert await distillation_lock.try_begin_run("ws", "mas") is True
    await distillation_lock.end_run("ws", "mas")


@pytest.mark.asyncio
async def test_different_keys_independent():
    assert await distillation_lock.try_begin_run("w1", "m1") is True
    assert await distillation_lock.try_begin_run("w2", "m1") is True
    await distillation_lock.end_run("w1", "m1")
    await distillation_lock.end_run("w2", "m1")
