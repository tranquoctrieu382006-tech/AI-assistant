"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager
from importlib.util import find_spec
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, rag
from app.core.config import settings
from app.core.llm import MODEL_REGISTRY, check_ollama_status_async
from app.core.tracing import init_tracing, flush_langfuse
from app.core.vector_store import close_pool
from app.core.security import RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_assistant")


def validate_optional_dependencies() -> None:
    requirements = []
    if settings.search_mode == "hybrid":
        requirements.append(("rank_bm25", "rank-bm25", "requirements.txt"))
    if settings.use_reranker:
        requirements.append(("sentence_transformers", "sentence-transformers", "requirements-reranker.txt"))

    missing = []
    for module, package, requirement_file in requirements:
        if find_spec(module) is None:
            missing.append(f"{package} (install with `pip install -r {requirement_file}`)")
    if missing:
        raise RuntimeError(
            "Enabled RAG feature is missing dependencies: " + "; ".join(missing)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup / shutdown hook."""
    validate_optional_dependencies()
    logger.info("Starting %s v%s...", settings.app_name, settings.app_version)

    # Khởi tạo tracing environment tường minh
    init_tracing()

    logger.info("Default model: %s", settings.default_model)
    logger.info("Configured roles: %s", list(MODEL_REGISTRY.keys()))

    ollama_info = await check_ollama_status_async()
    if ollama_info.get("status") == "ok":
        logger.info("Ollama is reachable. Available models: %s", ollama_info.get("models"))
    else:
        logger.warning("Ollama status check: %s", ollama_info)

    yield

    logger.info("Shutting down %s... Flushing remaining traces.", settings.app_name)
    flush_langfuse()
    close_pool()


app = FastAPI(
    title=settings.app_name,
    description="Mini AI Assistant – RAG & NLQ (FastAPI, LangChain, Langfuse, PostgreSQL/pgvector)",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware – sử dụng danh sách origin tường minh, an toàn khi có credentials
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(rag.router, prefix="/api", tags=["RAG"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": f"{settings.app_name} API is running",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
