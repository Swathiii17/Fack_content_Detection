from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AnalysisState
from config import settings
import httpx
import json

llm = ChatAnthropic(
    model=settings.model,
    api_key=settings.anthropic_api_key,
    max_tokens=2048,
)

SYSTEM_PROMPT = """You are a specialist fake news detection agent for investigative journalists.

Your job is to analyse news content for signs of misinformation, manipulation, or fabrication.

Evaluate:
1. SOURCE CREDIBILITY — is the source known, reputable, verifiable?
2. CLAIM VERIFICATION — are factual claims supported by evidence?
3. LANGUAGE PATTERNS — sensationalism, emotional manipulation, vague attribution
4. INTERNAL CONSISTENCY — contradictions, timeline issues
5. CONTEXT — missing context that changes meaning
6. WEB SEARCH RESULTS — do other credible sources corroborate or contradict?

Respond ONLY with valid JSON:
{
  "verdict": "fake" | "likely_fake" | "uncertain" | "likely_real" | "real",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation for the journalist",
  "signals": ["signal 1", "signal 2", ...],
  "source_score": 0.0-1.0,
  "claim_score": 0.0-1.0,
  "language_score": 0.0-1.0
}
"""

async def search_web(query: str) -> str:
    """Search using Tavily API for fact-checking."""
    if not settings.tavily_api_key:
        return "Web search unavailable (no API key configured)."
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                },
            )
            data = r.json()
            results = data.get("results", [])
            if not results:
                return "No search results found."
            summary = []
            for res in results[:4]:
                summary.append(f"- {res.get('title','')}: {res.get('content','')[:200]}")
            return "\n".join(summary)
    except Exception as e:
        return f"Search error: {str(e)}"


async def news_agent_node(state: AnalysisState) -> AnalysisState:
    """Detects fake news using LLM analysis + web search."""

    if "news" not in state.get("agents_to_run", []):
        return state

    content = ""
    if state.get("url"):
        content += f"URL: {state['url']}\n"
    if state.get("raw_text"):
        content += f"Article text:\n{state['raw_text'][:3000]}\n"

    if not content:
        return {**state, "news_result": {
            "verdict": "uncertain",
            "confidence": 0.0,
            "reasoning": "No content provided to analyse.",
            "signals": [],
            "source_score": 0.0,
            "claim_score": 0.0,
            "language_score": 0.0,
        }}

    # Extract key claim to fact-check
    claim_extract = await llm.ainvoke([
        SystemMessage(content="Extract the single most important factual claim from this content. Reply with just the claim in one sentence."),
        HumanMessage(content=content[:1000]),
    ])
    main_claim = claim_extract.content.strip()

    # Web search for corroboration
    search_results = await search_web(f"fact check: {main_claim}")

    user_msg = f"""Content to analyse:
{content}

Main claim identified: {main_claim}

Web search corroboration results:
{search_results}

Now analyse this content for fake news indicators."""

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
            "source_score": 0.5,
            "claim_score": 0.5,
            "language_score": 0.5,
        }

    result["agent"] = "news"
    return {**state, "news_result": result}
