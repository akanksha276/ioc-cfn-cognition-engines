# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""§7: at most one active distillation run per (workspace_id, mas_id)."""

from __future__ import annotations

import asyncio
from typing import Dict, Tuple

_Key = Tuple[str, str]

_lock_meta = asyncio.Lock()
_in_progress: Dict[_Key, bool] = {}


def _norm_key(workspace_id: str, mas_id: str) -> _Key:
    return (workspace_id.strip(), mas_id.strip())


async def try_begin_run(workspace_id: str, mas_id: str) -> bool:
    """
    Returns True if this call acquired the logical run slot (caller must end_run in all paths).
    Returns False if a run is already in progress → HTTP 409.
    """
    key = _norm_key(workspace_id, mas_id)
    async with _lock_meta:
        if _in_progress.get(key):
            return False
        _in_progress[key] = True
        return True


async def end_run(workspace_id: str, mas_id: str) -> None:
    key = _norm_key(workspace_id, mas_id)
    async with _lock_meta:
        _in_progress.pop(key, None)
