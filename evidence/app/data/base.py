# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Protocol


class DataRepository(Protocol):
    async def fetch_records(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        ...

    async def find_paths(
        self, source_id: str, target_id: str, max_depth: int, limit: int, relations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        ...

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        ...

