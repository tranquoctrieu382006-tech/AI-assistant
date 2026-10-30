from fastapi import APIRouter, Response, Security

from app.core.orchestrator import AskUnifiedRequest, AskUnifiedResponse, ask_orchestrated
from app.core.security import require_api_key

router = APIRouter(dependencies=[Security(require_api_key)])


@router.post("/ask", response_model=AskUnifiedResponse, summary="Điểm truy vấn hợp nhất (Auto / RAG / NLQ với Fallback an toàn)")
async def ask_endpoint(request: AskUnifiedRequest, response: Response = None) -> AskUnifiedResponse:
    result = await ask_orchestrated(request)
    if result.trace_id and response is not None:
        response.headers["X-Trace-Id"] = result.trace_id
    return result
