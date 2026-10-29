"""
Router API endpoint for intent classification.
Phân loại câu hỏi thành 'rag', 'nlq' hoặc 'out_of_scope'.
"""

import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter, Security

from app.core.security import require_api_key
from app.core.router import route_question, RouteCategory

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


class RouteRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="Câu hỏi cần phân loại")


class RouteResponse(BaseModel):
    category: RouteCategory
    reasoning: str


@router.post("/router/classify", response_model=RouteResponse, summary="Phân loại câu hỏi (RAG / NLQ / Out of scope)")
def classify_intent(request: RouteRequest) -> RouteResponse:
    decision = route_question(request.question)
    return RouteResponse(
        category=decision.category,
        reasoning=decision.reasoning,
    )
