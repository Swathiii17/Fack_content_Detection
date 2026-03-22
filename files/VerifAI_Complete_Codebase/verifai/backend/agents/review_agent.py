from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AnalysisState
from config import settings
import json
import re

llm = ChatAnthropic(
    model=settings.model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
)

SYSTEM_PROMPT = """You are a specialist fake review detection agent for journalists and researchers.

Analyse product/service reviews for signs of being fake, bot-generated, or incentivised.

Evaluate these signals:
1. LANGUAGE PATTERNS — generic praise, no specific details, unusually formal/informal
2. SENTIMENT ANOMALY — extreme positivity/negativity without nuance
3. SPECIFICITY — real reviews mention specific details (dates, product versions, use cases)
4. REVIEWER SIGNALS — brand new account, only 5-star reviews, burst posting
5. TEMPLATE PATTERNS — similar phrasing to other reviews (bot farms reuse templates)
6. AI GENERATION — repetitive structure, lack of personality, unnatural transitions
7. INCENTIVE SIGNALS — phrases like "I received this for free", "I was asked to review"

Respond ONLY with valid JSON:
{
  "verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "confidence": 0.0-1.0,
  "reasoning": "explanation for the journalist",
  "signals": ["signal 1", "signal 2", ...],
  "language_score": 0.0-1.0,
  "specificity_score": 0.0-1.0,
  "sentiment_score": 0.0-1.0,
  "ai_generated_probability": 0.0-1.0
}
"""

def extract_review_features(text: str) -> dict:
    """Extract simple heuristic features from review text."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    superlatives = sum(1 for w in words if w.lower() in [
        "best", "amazing", "perfect", "excellent", "worst", "terrible",
        "horrible", "outstanding", "fantastic", "incredible", "love", "hate"
    ])
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    exclamations = text.count("!")
    specific_refs = sum(1 for w in words if w.lower() in [
        "day", "week", "month", "year", "version", "model", "bought", "purchased",
        "arrived", "shipping", "color", "size", "quality"
    ])

    return {
        "word_count": len(words),
        "superlative_density": round(superlatives / max(len(words), 1), 3),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "exclamation_count": exclamations,
        "specific_reference_count": specific_refs,
    }


async def review_agent_node(state: AnalysisState) -> AnalysisState:
    """Detects fake reviews using LLM + heuristic analysis."""

    if "review" not in state.get("agents_to_run", []):
        return state

    text = state.get("raw_text", "")
    if not text:
        return {**state, "review_result": {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": "No review text provided.",
            "signals": [],
            "language_score": 0.0,
            "specificity_score": 0.0,
            "sentiment_score": 0.0,
            "ai_generated_probability": 0.0,
        }}

    features = extract_review_features(text)

    user_msg = f"""Review text to analyse:
\"\"\"{text[:2000]}\"\"\"

Heuristic features extracted:
- Word count: {features['word_count']}
- Superlative density: {features['superlative_density']} (higher = more suspicious)
- Average sentence length: {features['avg_sentence_length']} words
- Exclamation marks: {features['exclamation_count']}
- Specific references (dates, sizes, etc.): {features['specific_reference_count']}

Now perform your full fake review analysis."""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "verdict": "uncertain",
            "confidence": 0.5,
            "reasoning": response.content,
            "signals": [],
            "language_score": 0.5,
            "specificity_score": 0.5,
            "sentiment_score": 0.5,
            "ai_generated_probability": 0.5,
        }

    result["agent"] = "review"
    result["features"] = features
    return {**state, "review_result": result}
