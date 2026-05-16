"""
Main LangGraph graph.
 
Wires all agents together with routing, fan-out for travellers,
and two inline HITL interrupt checkpoints.
 
Flow:
  START
    → intent_classifier
    → OUT_OF_SCOPE: END
    → NEEDS_INFO:   elicitation (interrupt loop) → END
    → READY:        hitl_1
      → CORRECTED:  elicitation → hitl_1 (loop)
      → CONFIRMED:  memory_read
        → cache HIT:  hitl_2 → activities → END
        → cache MISS: orchestrator → memory_write → hitl_2 → activities → END
"""
 
import operator
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Send
from langgraph.checkpoint.memory import MemorySaver
 
from src.agents.intent_classifier import handle_user_query
from src.agents.orchestrator import run_orchestrator
from src.agents.memory_agent import run_memory_read, run_memory_write
from src.agents.activities import run_activities_agent
from src.state import (
    IntentStatus, HITLCheckpoint,
    HITLStatus, QueryState, query_state_from_dict, query_state_to_dict
)


# ── Node functions ────────────────────────────────────────────

def handle_user_query_node(state: dict) -> dict:
    return handle_user_query(state)

 
def memory_read_node(state: dict) -> dict:
    return run_memory_read(state)
 
 
def orchestrator_node(state: dict) -> dict:
    return run_orchestrator(state)
 
 
def memory_write_node(state: dict) -> dict:
    return run_memory_write(state)
 
 
def activities_node(state: dict) -> dict:
    return run_activities_agent(state)
 
 
def out_of_scope_node(state: dict) -> dict:
    """Returns a clean out-of-scope message."""
    reason = state.get("out_of_scope_reason", "")
    return {
        "agent_response": (
            f"That's outside what I can help with — {reason}.\n\n"
            f"I help groups of 1–5 people flying from different African or "
            f"European cities find the cheapest destination to meet, "
            f"including flights, accommodation and things to do."
        )
    }
 

 # ── HITL Checkpoint 1 — pre-search confirmation ───────────────
 
def hitl_1_node(state: dict) -> dict:
    """
    Builds confirmation message, calls interrupt() to pause.
    Resumes when CLI sends Command(resume=user_input).
    """
    query_state = query_state_from_dict(state.get("query_state", {}))
 
    traveller_lines = "\n".join(
        f"  - {t.name}: flying from {t.origin_city} ({t.origin_iata})"
        for t in query_state.travellers
    )
    region = query_state.region_preferences or "anywhere"
 
    message = (
        f"Here's what I have:\n"
        f"{traveller_lines}\n"
        f"  - Month: {query_state.travel_month}\n"
        f"  - Duration: {query_state.duration_nights} nights\n"
        f"  - Region: {region}\n\n"
        f"Is that correct? (yes / no)"
    )
 
    # Pause here — CLI prints message, user responds, CLI resumes with Command
    user_response = interrupt(message)
 
    confirmed = user_response.strip().lower() in ("yes", "y", "correct", "yep", "yeah", "ok", "sure", "looks good", "correct")
    print(f"User response to HITL 1: '{user_response}' → confirmed={confirmed}")
 
    checkpoint = HITLCheckpoint(
        checkpoint_id="hitl_1",
        message=message,
        status=HITLStatus.CONFIRMED if confirmed else HITLStatus.CORRECTED,
        user_response=user_response,
        correction=None if confirmed else {"raw": user_response},
    )
 
    return {
        "hitl_checkpoint_1": checkpoint,
        "hitl_1_confirmed": confirmed,
        "elicitation_complete": confirmed,
        "query_state": state.get("query_state"),
    }
    
# ── HITL Checkpoint 2 — post-ranking destination selection ────
 
def hitl_2_node(state: dict) -> dict:
    """
    Interrupt node — shows ranked destinations and asks which
    ones to fetch activities for.
    """
    ranked = state.get("ranked_destinations", [])
    print(f"[hitl_2] ranked_destinations count: {len(ranked)}, first: {ranked[0] if ranked else 'EMPTY'}")  # ← here
 
    if not ranked:
        return {
            "hitl_checkpoint_2": None,
            "activity_destinations": [],
            "awaiting_hitl": False,
        }
 
    top3 = ranked[:3]
    dest_lines = []
    for dest in top3:
        total = dest.grand_total_usd or 0.0
        breakdown = ""
        if dest.per_traveller_breakdown:
            parts = [
                f"{b.traveller_name}: {b.total_local:,.0f} {b.currency}"
                for b in dest.per_traveller_breakdown
            ]
            breakdown = " | ".join(parts)
        dest_lines.append(
            f"  #{dest.rank} {dest.city} ({dest.iata}) — "
            f"${total:,.0f} total"
            + (f"\n       {breakdown}" if breakdown else "")
        )
 
    dest_summary = "\n".join(dest_lines)
    options = "  1 — Just #1\n"
    if len(top3) >= 2:
        options += "  2 — #1 and #2\n"
    if len(top3) >= 3:
        options += "  3 — All three\n"
 
    message = (
        f"Top destinations:\n{dest_summary}\n\n"
        f"Get activities for:\n{options}"
        f"Enter 1, 2, or 3:"
    )
 
    checkpoint = HITLCheckpoint(
        checkpoint_id="hitl_2",
        message=message,
        status=HITLStatus.PENDING,
    )
 
    user_response = interrupt(message)
    checkpoint.user_response = user_response
    checkpoint.status = HITLStatus.CONFIRMED
 
    choice = user_response.strip()
    if choice == "1":
        activity_destinations = [top3[0].iata]
    elif choice == "2" and len(top3) >= 2:
        activity_destinations = [top3[0].iata, top3[1].iata]
    else:
        activity_destinations = [d.iata for d in top3]
 
    return {
        "hitl_checkpoint_2": checkpoint,
        "activity_destinations": activity_destinations,
        "awaiting_hitl": False,
        "query_state": state.get("query_state"),
    }



# ── Conditional routing ───────────────────────────────────────

def route_user_query(state: dict) -> Literal["out_of_scope", "hitl_1", "end"]:
    if state.get("intent_status") == IntentStatus.OUT_OF_SCOPE:
        return "out_of_scope"
    if state.get("elicitation_complete"):
        return "hitl_1"
    return "end"
 
def route_hitl_1(state: dict) -> Literal["user_query", "memory_read"]:
    cp = state.get("hitl_checkpoint_1")
    print(f"\n Routing HITL 1 with state: {state} on route_hitl_1 and corrected is {cp.status == HITLStatus.CORRECTED if cp else False}\n") 
    if cp and cp.status == HITLStatus.CORRECTED:
        return "user_query"
    return "memory_read"
 
 
def route_memory_read(state: dict) -> Literal["hitl_2", "orchestrator"]:
    if state.get("episodic_cache_hit"):
        return "hitl_2"
    return "orchestrator"
 
 
# ── Fan-out: one traveller node per traveller ─────────────────
 
def fan_out_travellers(state: dict) -> list[Send]:
    """
    Dynamically creates one Send per traveller for parallel execution.
    Each traveller node runs run_traveller_agent independently.
    Results are merged via operator.add on flight_results.
    NOTE: This is used when the orchestrator is split into separate
    graph nodes (advanced wiring). For MVP the orchestrator handles
    fan-out internally. This function is provided for Step 14 extension.
    """
    from src.agents.traveller import run_traveller_agent
    query_state = query_state_from_dict(state.get("query_state", {}))
    return [
        Send("traveller_node", {**state, "current_traveller": t})
        for t in query_state.travellers
    ]
 
 

# ── Build the main graph ──────────────────────────────────────
 
def build_graph():
    graph = StateGraph(dict)
 
    # Add all nodes
    graph.add_node("user_query", handle_user_query_node)
    graph.add_node("out_of_scope",      out_of_scope_node)
    graph.add_node("hitl_1",            hitl_1_node)
    graph.add_node("memory_read",       memory_read_node)
    graph.add_node("orchestrator",      orchestrator_node)
    graph.add_node("memory_write",      memory_write_node)
    graph.add_node("hitl_2",            hitl_2_node)
    graph.add_node("activities",        activities_node)
 
    # Entry point
    graph.add_edge(START, "user_query")
 
    # Intent routing
    graph.add_conditional_edges(
        "user_query",
        route_user_query,
        {
            "out_of_scope": "out_of_scope",
            "hitl_1":       "hitl_1",
            "end":           END,
        },
    )

 
    # Out of scope → END
    graph.add_edge("out_of_scope", END)
 
    # HITL 1 routing
    graph.add_conditional_edges(
        "hitl_1",
        route_hitl_1,
        {
            "user_query":  "user_query",
            "memory_read":  "memory_read",
        },
    )
 
    # Memory read routing
    graph.add_conditional_edges(
        "memory_read",
        route_memory_read,
        {
            "hitl_2":      "hitl_2",
            "orchestrator": "orchestrator",
        },
    )
 
    # Search path
    graph.add_edge("orchestrator",  "memory_write")
    graph.add_edge("memory_write",  "hitl_2")
 
    # HITL 2 → activities → END
    graph.add_edge("hitl_2",       "activities")
    graph.add_edge("activities",   END)
 
    return graph.compile(checkpointer=MemorySaver())
 
 
# Module-level compiled graph
travel_agent_graph = build_graph()
 
try:
    travel_agent_graph.get_graph().draw_mermaid_png(
        output_file_path="travel_agent_graph.png"
    )
    print("\nGraph saved as travel_agent_graph.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")