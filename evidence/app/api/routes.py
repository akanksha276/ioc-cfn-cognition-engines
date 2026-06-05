# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Request

from ..agent.evidence_ce import EvidenceAction, EvidenceCognitionEngine
from ..dependencies import (
    get_evidence_cognition_engine,
    get_repository_for_reasoning,
)
from .schemas import (
    Concept,
    ConceptsByIdsRequest,
    ConceptsByIdsResponse,
    GraphPathsRequest,
    GraphPathsResponse,
    NeighborsResponse,
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
)

router = APIRouter()


@router.post(
    "/reasoning/evidence",
    response_model=ReasonerCognitionResponse,
    response_model_exclude_none=True,
)
async def reasoning_evidence(
    req: ReasonerCognitionRequest,
    request: Request,
    engine: EvidenceCognitionEngine = Depends(get_evidence_cognition_engine),
):
    # Scope the engine's repo to the request's workspace/MAS for correct CFN path resolution.
    engine._repo = get_repository_for_reasoning(request, req)
    return await engine.run(EvidenceAction.REASON, req.model_dump(mode="json"))


# ---- DB-facing endpoints routed through EvidenceCognitionEngine ----

@router.post("/graph/paths", response_model=GraphPathsResponse)
async def graph_paths(
    req: GraphPathsRequest,
    engine: EvidenceCognitionEngine = Depends(get_evidence_cognition_engine),
):
    result = await engine.run(EvidenceAction.GRAPH_PATHS, req.model_dump(mode="json"))
    return GraphPathsResponse(
        status=result.get("status", "success"),
        paths=result.get("paths", []),
    )


@router.get("/graph/neighbors/{concept_id}", response_model=NeighborsResponse)
async def graph_neighbors(
    concept_id: str,
    engine: EvidenceCognitionEngine = Depends(get_evidence_cognition_engine),
):
    result = await engine.run(EvidenceAction.NEIGHBORS, {"node_id": concept_id})
    return NeighborsResponse(records=result.get("records", []))


@router.post("/graph/concepts/by_ids", response_model=ConceptsByIdsResponse)
async def graph_concepts_by_ids(
    req: ConceptsByIdsRequest,
    engine: EvidenceCognitionEngine = Depends(get_evidence_cognition_engine),
):
    result = await engine.run(EvidenceAction.CONCEPTS_BY_IDS, req.model_dump(mode="json"))
    concepts = [
        Concept(
            id=str(c.get("id", "")),
            name=str(c.get("name", "")),
            type=str(c.get("type", "")),
            description=str(c.get("description", "")),
        )
        for c in result.get("concepts", [])
    ]
    return ConceptsByIdsResponse(concepts=concepts)

