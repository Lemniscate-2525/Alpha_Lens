"""
AlphaLens - Query Endpoint
Receives financial user queries and returns audited synthesis with citations and metrics.
"""
from fastapi import APIRouter, Depends, Request
from src.core.models import QueryRequest, QueryResponse
from src.services.rag_service import rag_service
from src.api.dependencies import verify_api_key

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    request_id = getattr(raw_request.state, "request_id", None)
    response = await rag_service.execute_query(request, trace_id=request_id)
    return response

