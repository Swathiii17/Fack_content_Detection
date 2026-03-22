# ─────────────────────────────────────────────────────────────────
# REPLACE backend/models/state.py with this updated version
# Adds video_path and video_result to AnalysisState
# ─────────────────────────────────────────────────────────────────
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel


class AgentResult(BaseModel):
    agent: str
    verdict: Literal["fake", "likely_fake", "uncertain", "likely_real", "real"]
    confidence: float
    reasoning: str
    signals: list[str]


class AnalysisState(TypedDict):
    # Input
    content_type: str
    raw_text: Optional[str]
    url: Optional[str]
    image_path: Optional[str]
    audio_path: Optional[str]
    video_path: Optional[str]          # NEW
    filename: Optional[str]

    # Intermediate results per agent
    news_result: Optional[dict]
    review_result: Optional[dict]
    image_result: Optional[dict]
    audio_result: Optional[dict]
    video_result: Optional[dict]       # NEW

    # Orchestrator decision
    agents_to_run: list[str]

    # Final
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    all_signals: list[str]
    error: Optional[str]
