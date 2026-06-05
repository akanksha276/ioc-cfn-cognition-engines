# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional


class MockDataRepository:
    async def fetch_records(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"id": 1, "value": 42}, {"id": 2, "value": 99}]

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "node": {"id": concept_id, "name": concept_id},
                    "relationships": [],
                    "neighbors": [],
                }
            ]
        }

    async def find_paths(
        self, source_id: str, target_id: str, max_depth: int, limit: int, relations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "paths": [
                {
                    "node_ids": [source_id, target_id],
                    "edges": [
                        {"from_id": source_id, "to_id": target_id, "relation": "RELATED_TO"},
                    ],
                    "path_length": 1,
                    "symbolic": f"{source_id} -RELATED_TO-> {target_id}",
                }
            ],
        }

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        return [{"id": i, "name": i, "description": "", "type": "concept"} for i in ids or []]
