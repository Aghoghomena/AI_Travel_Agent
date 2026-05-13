"""
Ranker Agent.
 
Receives all DestinationResults with USD costs already normalised.
Uses chain-of-thought to rank destinations cheapest first.
Falls back to simple sort if LLM output is malformed.
 
LangGraph node: run_ranker_agent(state) -> dict
"""

import json
from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.prompts.ranker import build_ranker_prompt
from src.utils.config import llm
from src.state import RankerState

# ── Tools ─────────────────────────────────────────────────────
 
def sort_by_total(destination_results: list) -> list:
    """Simple sort fallback — cheapest grand_total_usd first."""
    return sorted(
        destination_results,
        key=lambda d: d.grand_total_usd or float("inf"),
    )


def apply_ranking(destination_results: list, ranking: list[dict]) -> list:
    """
    Applies LLM ranking to destination results.
    Matches by iata code, assigns rank field.
    """
    rank_map = {r["iata"]: r["rank"] for r in ranking}
    for dest in destination_results:
        dest.rank = rank_map.get(dest.iata, 99)
    return sorted(destination_results, key=lambda d: d.rank or 99)


def validate_ranking(parsed: list, destination_results: list) -> tuple[bool, str]:
    """Validates LLM ranking output."""
    if not isinstance(parsed, list):
        return False, "Output is not a list"
    if len(parsed) != len(destination_results):
        return False, f"Expected {len(destination_results)} items, got {len(parsed)}"
    ranks = [r.get("rank") for r in parsed]
    iatas = [r.get("iata") for r in parsed]
    if sorted(ranks) != list(range(1, len(parsed) + 1)):
        return False, f"Ranks are not sequential 1..n: {ranks}"
    expected_iatas = {d.iata for d in destination_results}
    if set(iatas) != expected_iatas:
        return False, f"IATA mismatch: got {iatas}, expected {expected_iatas}"
    return True, ""

# ── Nodes ─────────────────────────────────────────────────────
 
def rank_node(state: RankerState) -> RankerState:
    """
    Node 1: Calls LLM with chain-of-thought to rank destinations.
    """
    destination_results = state.get("destination_results", [])
    prompt = build_ranker_prompt(destination_results)
    response = llm.invoke(prompt)
    raw = response.content.strip()
 
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
 
    return {**state, "raw_llm_output": raw}
 
 
def validate_node(state: RankerState) -> RankerState:
    """
    Node 2: Validates the LLM ranking output.
    """
    raw = state.get("raw_llm_output", "")
 
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {**state, "parsed_ranking": None, "error": f"JSON parse failed: {e}"}
 
    is_valid, error_msg = validate_ranking(parsed, state.get("destination_results", []))
    if not is_valid:
        return {**state, "parsed_ranking": None, "error": error_msg}
 
    return {**state, "parsed_ranking": parsed, "error": None}
 
 
def retry_node(state: RankerState) -> RankerState:
    """
    Node 3: Increments retry counter.
    On max retries falls back to simple sort.
    """
    retry_count = state.get("retry_count", 0) + 1
 
    if retry_count >= 2:
        # Fallback — sort by grand_total_usd directly
        sorted_results = sort_by_total(state.get("destination_results", []))
        for i, dest in enumerate(sorted_results):
            dest.rank = i + 1
        fallback_ranking = [
            {"iata": d.iata, "grand_total_usd": d.grand_total_usd, "rank": d.rank}
            for d in sorted_results
        ]
        return {
            **state,
            "retry_count": retry_count,
            "parsed_ranking": fallback_ranking,
            "error": None,
        }
 
    return {**state, "retry_count": retry_count}
 
 
def apply_ranking_node(state: RankerState) -> RankerState:
    """
    Node 4: Applies ranking to destination results.
    Returns ranked_destinations sorted cheapest first.
    """
    ranked = apply_ranking(
        state.get("destination_results", []),
        state.get("parsed_ranking", []),
    )
    return {**state, "ranked_destinations": ranked}

 
# ── Conditional edges ─────────────────────────────────────────
 
def should_retry(state: RankerState) -> Literal["retry", "apply"]:
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    return "apply"
 
 
def retry_or_apply(state: RankerState) -> Literal["rank", "apply"]:
    if state.get("parsed_ranking"):
        return "apply"
    return "rank"


# ── Build the agent graph ─────────────────────────────────────
 
def build_ranker_graph():
    graph = StateGraph(RankerState)
 
    graph.add_node("rank", rank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("apply_ranking", apply_ranking_node)
 
    graph.add_edge(START, "rank")
    graph.add_edge("rank", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"retry": "retry", "apply": "apply_ranking"},
    )
    graph.add_conditional_edges(
        "retry",
        retry_or_apply,
        {"rank": "rank", "apply": "apply_ranking"},
    )
    graph.add_edge("apply_ranking", END)
 
    return graph.compile()
 
 
ranker_agent = build_ranker_graph()
try:
    ranker_agent.get_graph().draw_mermaid_png(output_file_path="ranker_agent.png")
    print("\nGraph saved as ranker_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")
 
 
# ── Entry point for main travel agent graph ───────────────────
 
def run_ranker_agent(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Ranks destination_results cheapest first.
    """
    result = ranker_agent.invoke({
        "destination_results": state.get("destination_results", []),
        "raw_llm_output": None,
        "parsed_ranking": None,
        "retry_count": 0,
        "error": None,
        "ranked_destinations": [],
    })
 
    return {
        "ranked_destinations": result.get("ranked_destinations", []),
    }