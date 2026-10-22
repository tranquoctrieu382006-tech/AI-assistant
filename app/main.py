"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health
from app.core.config import settings
from app.core.llm import MODEL_REGISTRY, check_ollama_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup / shutdown hook."""
    logger.info("Starting AI Assistant API v0.1.0...")
    logger.info("Default model: %s", settings.default_model)
    logger.info("Configured roles: %s", list(MODEL_REGISTRY.keys()))

    ollama_info = check_ollama_status()
    if ollama_info.get("status") == "ok":
        logger.info("Ollama is reachable. Available models: %s", ollama_info.get("models"))
    else:
        logger.warning("Ollama status check: %s", ollama_info)

    yield

    logger.info("Shutting down AI Assistant API...")


app = FastAPI(
    title="AI Assistant",
    description="Mini AI Assistant – RAG & NLQ (FastAPI, LangChain, Langfuse, PostgreSQL/pgvector)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api", tags=["Health"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI Assistant API is running",
        "docs": "/docs",
        "health": "/api/health",
    }
