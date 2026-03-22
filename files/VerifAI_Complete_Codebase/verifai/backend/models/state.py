from typing import TypedDict, Optional, Literal
from pydantic import BaseModel


class AgentResult(BaseModel):
    agent: str
    verdict: Literal["fake", "likely_fake", "uncertain", "likely_real", "real"]
    confidence: float  # 0.0 - 1.0
    reasoning: str
    signals: list[str]  # key evidence points


class AnalysisState(TypedDict):
    # Input
    content_type: str           # "news" | "review" | "image" | "audio"
    raw_text: Optional[str]
    url: Optional[str]
    image_path: Optional[str]
    audio_path: Optional[str]
    filename: Optional[str]

    # Intermediate results per agent
    news_result: Optional[dict]
    review_result: Optional[dict]
    image_result: Optional[dict]
    audio_result: Optional[dict]

    # Orchestrator decision
    agents_to_run: list[str]

    # Final
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    all_signals: list[str]
    error: Optional[str]
