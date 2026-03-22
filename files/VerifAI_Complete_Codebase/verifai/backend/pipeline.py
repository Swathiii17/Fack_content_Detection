from langgraph.graph import StateGraph, END
from models.state import AnalysisState
from agents.orchestrator import orchestrator_node
from agents.news_agent import news_agent_node
from agents.review_agent import review_agent_node
from agents.image_agent import image_agent_node
from agents.audio_agent import audio_agent_node
from agents.verdict_aggregator import verdict_aggregator_node
import asyncio


def should_run_news(state: AnalysisState) -> bool:
    return "news" in state.get("agents_to_run", [])

def should_run_review(state: AnalysisState) -> bool:
    return "review" in state.get("agents_to_run", [])

def should_run_image(state: AnalysisState) -> bool:
    return "image" in state.get("agents_to_run", [])

def should_run_audio(state: AnalysisState) -> bool:
    return "audio" in state.get("agents_to_run", [])


async def parallel_agents_node(state: AnalysisState) -> AnalysisState:
    """
    Run all relevant specialist agents in parallel.
    Each agent checks internally if it should run.
    """
    tasks = []
    agents_to_run = state.get("agents_to_run", [])

    if "news" in agents_to_run:
        tasks.append(news_agent_node(state))
    if "review" in agents_to_run:
        tasks.append(review_agent_node(state))
    if "image" in agents_to_run:
        tasks.append(image_agent_node(state))
    if "audio" in agents_to_run:
        tasks.append(audio_agent_node(state))

    if not tasks:
        return state

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged = dict(state)
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, dict):
            for key in ["news_result", "review_result", "image_result", "audio_result"]:
                if key in result and result[key] is not None:
                    merged[key] = result[key]

    return merged


def build_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("specialist_agents", parallel_agents_node)
    graph.add_node("verdict_aggregator", verdict_aggregator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "specialist_agents")
    graph.add_edge("specialist_agents", "verdict_aggregator")
    graph.add_edge("verdict_aggregator", END)

    return graph.compile()


# Compiled graph — import this in the API
verifai_graph = build_graph()
