"""
Activities Agent.
 
Runs ONLY on the top 3 destinations after ranking (Step 10).
Checks semantic memory cache first. On cache miss calls Google Places.
Returns top 3 activities per destination sorted by rating.
 
LangGraph node: run_activities_agent(state) -> dict
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.tools.google_places import get_activities
from src.state import Activity, ActivitiesState

 
# ── Tools ─────────────────────────────────────────────────────
 
def check_semantic_cache(semantic_memory, destination_iata: str) -> list[Activity] | None:
    """Checks semantic memory for cached activities. No-op until Step 11."""
    if semantic_memory is None:
        return None
    try:
        cached = semantic_memory.get_activities(destination_iata)
        return cached if cached else None
    except Exception:
        return None
    

def write_semantic_cache(semantic_memory, destination_iata: str, activities: list[Activity]) -> None:
    """Writes activities to semantic memory. No-op until Step 11."""
    if semantic_memory is None:
        return
    try:
        semantic_memory.set_activities(destination_iata, activities)
    except Exception:
        pass

# ── Nodes ─────────────────────────────────────────────────────
 
def fetch_activities_node(state: ActivitiesState) -> ActivitiesState:
    """
    Node 1: Fetches top 3 activities for each destination.
    Checks semantic cache first. On miss calls Google Places API.
    Writes results to cache.
    """
    destination_iatas = state.get("destination_iatas", [])
    semantic_memory = state.get("semantic_memory")
    activities = dict(state.get("activities", {}))
    errors = list(state.get("errors", []))
 
    for iata in destination_iatas:
        # Check semantic cache first
        cached = check_semantic_cache(semantic_memory, iata)
        if cached:
            activities[iata] = cached
            continue
 
        # Cache miss — call Google Places
        results = get_activities(iata)
 
        if results:
            write_semantic_cache(semantic_memory, iata, results)
            activities[iata] = results
        else:
            errors.append(f"No activities found for {iata}")
            activities[iata] = []
 
    return {
        **state,
        "activities": activities,
        "errors": errors,
    }


# ── Build the agent graph ─────────────────────────────────────
 
def build_activities_graph():
    graph = StateGraph(ActivitiesState)
 
    graph.add_node("fetch_activities", fetch_activities_node)
 
    graph.add_edge(START, "fetch_activities")
    graph.add_edge("fetch_activities", END)
 
    return graph.compile()
 
 
activities_agent = build_activities_graph()
try:
    activities_agent.get_graph().draw_mermaid_png(output_file_path="activities_agent.png")
    print("\nGraph saved as activities_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")
 
 
# ── Entry point for main travel agent graph ───────────────────
 
def run_activities_agent(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Called after HITL checkpoint 2 — only runs on confirmed destinations.
    """
    # activity_destinations set by HITL checkpoint 2 (Step 13)
    destination_iatas = state.get("activity_destinations", [])
 
    result = activities_agent.invoke({
        "destination_iatas": destination_iatas,
        "activities": {},
        "semantic_memory": state.get("semantic_memory"),
        "errors": [],
    })
 
    return {
        "activities": result.get("activities", {}),
        "errors": result.get("errors", []),
    }
 
 
