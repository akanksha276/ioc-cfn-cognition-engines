# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends
from .schemas import (
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
    GraphPathsRequest,
    GraphPathsResponse,
    NeighborsResponse,
    ConceptsByIdsRequest,
    ConceptsByIdsResponse,
    Concept,
)
from ..dependencies import (
    get_repository,
    get_repository_for_reasoning,
)
from ..agent.evidence import process_evidence

router = APIRouter()


@router.post(
    "/reasoning/evidence",
    response_model=ReasonerCognitionResponse,
    response_model_exclude_none=True,
)
async def reasoning_evidence(
    req: ReasonerCognitionRequest,
    repo=Depends(get_repository_for_reasoning),
):
    return await process_evidence(
        req,
        repo_adapter=repo,
    )


# ---- Placeholder DB-facing endpoints (wired to repository) ----

@router.post("/graph/paths", response_model=GraphPathsResponse)
async def graph_paths(req: GraphPathsRequest, repo=Depends(get_repository)):
    result = await repo.find_paths(
        source_id=req.source_id,
        target_id=req.target_id,
        max_depth=req.max_depth,
        limit=req.limit,
        relations=req.relations,
    )
    # Assume repo returns keys compatible with GraphPathsResponse
    return GraphPathsResponse(status=result.get("status", "success"), paths=result.get("paths", []))


@router.get("/graph/neighbors/{concept_id}", response_model=NeighborsResponse)
async def graph_neighbors(concept_id: str, repo=Depends(get_repository)):
    result = await repo.neighbors(concept_id)
    return NeighborsResponse(records=result.get("records", []))


@router.post("/graph/concepts/by_ids", response_model=ConceptsByIdsResponse)
async def graph_concepts_by_ids(req: ConceptsByIdsRequest, repo=Depends(get_repository)):
    rows = await repo.get_concepts_by_ids(req.ids)
    concepts = [
        Concept(
            id=str(row.get("id", "")),
            name=str(row.get("name", "")),
            type=str(row.get("type", "")),
            description=str(row.get("description", "")),
        )
        for row in (rows or [])
    ]
    return ConceptsByIdsResponse(concepts=concepts)

