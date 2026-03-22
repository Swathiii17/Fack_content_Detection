from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AnalysisState
from config import settings
import json

llm = ChatAnthropic(
    model=settings.model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
)

VERDICT_WEIGHTS = {
    "fake": 0.0,
    "likely_fake": 0.25,
    "uncertain": 0.5,
    "likely_real": 0.75,
    "real": 1.0,
}

SYSTEM_PROMPT = """You are the final verdict aggregator for a fake content detection system used by journalists.

You receive results from multiple specialist agents and must synthesise them into a single clear verdict.

Your job:
1. Weigh agent results based on relevance and confidence
2. Identify the strongest evidence signals
3. Produce a final verdict with clear reasoning a journalist can act on
4. Flag any contradictions between agents

Respond ONLY with valid JSON:
{
  "final_verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "final_confidence": 0.0-1.0,
  "final_reasoning": "Clear, journalist-actionable summary of findings",
  "all_signals": ["most important signal 1", "signal 2", ...],
  "contradictions": "any contradictions between agents, or null"
}
"""

async def verdict_aggregator_node(state: AnalysisState) -> AnalysisState:
    """Aggregates results from all specialist agents into a final verdict."""

    agent_results = {}
    all_signals = []

    for key in ["news_result", "review_result", "image_result", "audio_result"]:
        result = state.get(key)
        if result:
            agent_name = key.replace("_result", "")
            agent_results[agent_name] = result
            signals = result.get("signals", [])
            all_signals.extend(signals)

    if not agent_results:
        return {**state,
                "final_verdict": "uncertain",
                "final_confidence": 0.0,
                "final_reasoning": "No agents produced results.",
                "all_signals": []}

    # Simple weighted average as baseline
    weighted_scores = []
    for name, result in agent_results.items():
        verdict = result.get("verdict", "uncertain")
        confidence = result.get("confidence", 0.5)
        score = VERDICT_WEIGHTS.get(verdict, 0.5)
        weighted_scores.append(score * confidence)

    baseline_score = sum(weighted_scores) / len(weighted_scores)

    summary_parts = []
    for name, result in agent_results.items():
        summary_parts.append(
            f"[{name.upper()} AGENT]\n"
            f"Verdict: {result.get('verdict')} (confidence: {result.get('confidence', 0):.0%})\n"
            f"Reasoning: {result.get('reasoning', 'N/A')[:400]}\n"
            f"Signals: {', '.join(result.get('signals', [])[:4])}"
        )

    user_msg = f"""Agent results to synthesise:

{chr(10).join(summary_parts)}

Baseline weighted authenticity score: {baseline_score:.2f} (0=fake, 1=real)

Synthesise these into a final verdict for the journalist."""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        result = json.loads(response.content)
        return {
            **state,
            "final_verdict": result.get("final_verdict", "uncertain"),
            "final_confidence": result.get("final_confidence", 0.5),
            "final_reasoning": result.get("final_reasoning", ""),
            "all_signals": result.get("all_signals", all_signals[:8]),
        }
    except Exception:
        # Fallback to simple rule-based verdict
        if baseline_score < 0.25:
            verdict = "fake"
        elif baseline_score < 0.4:
            verdict = "likely_fake"
        elif baseline_score < 0.6:
            verdict = "uncertain"
        elif baseline_score < 0.75:
            verdict = "likely_real"
        else:
            verdict = "real"

        return {
            **state,
            "final_verdict": verdict,
            "final_confidence": abs(baseline_score - 0.5) * 2,
            "final_reasoning": response.content[:500],
            "all_signals": all_signals[:8],
        }
