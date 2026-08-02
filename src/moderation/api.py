"""FastAPI service wrapping the moderation cascade.

    uvicorn moderation.api:app --reload

Models are loaded once at startup, not per request, a cold `from_pretrained`
costs tens of seconds and would dominate every response.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import ModerationPipeline

logger = logging.getLogger(__name__)

_state: dict[str, ModerationPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading moderation models (this takes a moment)")
    pipeline = ModerationPipeline()
    # Force both models to load now, so the first real request is fast and a
    # broken model config fails at boot rather than mid-traffic.
    pipeline.hate_clf("تجربة")
    pipeline.toxicity_clf("تجربة")
    _state["pipeline"] = pipeline
    logger.info("Models ready")
    yield
    _state.clear()


app = FastAPI(
    title="Arabic Content Moderation API",
    description=(
        "Two-stage moderation cascade for Egyptian-dialect Arabic. Returns a category, "
        "a recommended action, and an Arabic explanation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, examples=["انت غبي"])


class BatchModerateRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class ModerateResponse(BaseModel):
    text: str
    category: str
    confidence: float
    stage: str
    action: str
    escalated: bool
    explanation: str
    masked_text: str | None = None
    is_harmful: bool


def _to_response(verdict) -> ModerateResponse:
    return ModerateResponse(**verdict.as_dict(), is_harmful=verdict.is_harmful)


def _pipeline() -> ModerationPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    return pipeline


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "pipeline" in _state else "loading",
        "hate_model": settings.hate_model,
        "toxicity_model": settings.toxicity_model,
    }


@app.post("/moderate", response_model=ModerateResponse)
def moderate(request: ModerateRequest) -> ModerateResponse:
    """Moderate a single piece of text."""
    return _to_response(_pipeline().moderate(request.text))


@app.post("/moderate/batch", response_model=list[ModerateResponse])
def moderate_batch(request: BatchModerateRequest) -> list[ModerateResponse]:
    """Moderate up to 64 texts in one call."""
    verdicts = _pipeline().moderate_batch(request.texts)
    return [_to_response(v) for v in verdicts]
