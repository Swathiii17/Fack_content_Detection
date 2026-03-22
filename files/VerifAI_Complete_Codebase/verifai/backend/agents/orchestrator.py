from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AnalysisState
from config import settings
import json

llm = ChatAnthropic(
    model=settings.model,
    api_key=settings.anthropic_api_key,
    max_tokens=1024,
)

SYSTEM_PROMPT = """You are an orchestrator for a fake content detection system used by journalists.

Your job is to decide which specialist agents to run based on the content provided.

Available agents:
- "news": For news articles, claims, URLs to news sites
- "review": For product/service reviews, testimonials  
- "image": For image files or image URLs
- "audio": For audio files or voice recordings

Respond ONLY with valid JSON:
{
  "agents_to_run": ["agent1", "agent2"],
  "reasoning": "brief explanation"
}

Rules:
- Always include at least one agent
- For URLs: guess content type from URL structure
- For plain text without context: run both "news" and "review"
- For mixed content: run multiple agents
"""

async def orchestrator_node(state: AnalysisState) -> AnalysisState:
    """Decides which agents to invoke based on input."""

    user_msg = f"Content type hint: {state.get('content_type', 'unknown')}\n"
    if state.get("url"):
        user_msg += f"URL: {state['url']}\n"
    if state.get("raw_text"):
        user_msg += f"Text preview: {state['raw_text'][:500]}\n"
    if state.get("image_path"):
        user_msg += "Has image file attached.\n"
    if state.get("audio_path"):
        user_msg += "Has audio file attached.\n"
    if state.get("filename"):
        user_msg += f"Filename: {state['filename']}\n"

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    try:
        parsed = json.loads(response.content)
        agents_to_run = parsed.get("agents_to_run", [state.get("content_type", "news")])
    except Exception:
        agents_to_run = [state.get("content_type", "news")]

    return {**state, "agents_to_run": agents_to_run}
