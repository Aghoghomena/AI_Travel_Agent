"""
Main LangGraph graph.

Wires all agents together with routing and two inline HITL interrupt checkpoints.

Flow:
  START
    → user_query
    → OUT_OF_SCOPE: END
    → NEEDS_INFO:   elicitation (interrupt loop) → END
    → READY:        hitl_1
      → CORRECTED:  user_query (loop)
      → CONFIRMED:  plan
          → validate_plan → [replan →] execute → hitl_2 → activities → END

The ReWOO plan decides which tools to run (memory_read, traveller_agent,
accommodation_agent, currency_agent, ranker_agent, memory_write).
execute_node follows the plan exactly.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from src.agents.intent_classifier import handle_user_query
from src.agents.orchestrator import (
    plan_node, validate_plan_node, replan_node, execute_node,
    plan_valid, replan_or_execute,
)
from src.agents.activities import run_activities_agent
from src.state import (
    IntentStatus, HITLCheckpoint,
    HITLStatus, QueryState, query_state_from_dict, query_state_to_dict,TravelAgentState
)


# ── Node functions ────────────────────────────────────────────

def handle_user_query_node(state: dict) -> dict:
    result = handle_user_query(state)
    intent = result.get("intent_status")
    return {
        "intent_status":        intent.value if isinstance(intent, IntentStatus) else intent,
        "query_state":          result.get("query_state"),
        "agent_response":       result.get("agent_response"),
        "elicitation_question": result.get("elicitation_question"),
        "elicitation_complete": result.get("elicitation_complete", False),
        "turn_limit_reached":   result.get("turn_limit_reached", False),
        "conversation_history": result.get("conversation_history", []),
        "turn_count":           result.get("turn_count", 0),
    }



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
        "ranked_destinations":   ranked,
        "query_state": state.get("query_state"),
    }



# ── Conditional routing ───────────────────────────────────────

def route_user_query(state: dict) -> Literal["out_of_scope", "hitl_1", "end"]:
    print(f"\n state at 184 {state} \n")
    if state.get("intent_status") == IntentStatus.OUT_OF_SCOPE:
        return "out_of_scope"
    if state.get("elicitation_complete"):
        return "hitl_1"
    return "end"


def route_hitl_1(state: dict) -> Literal["user_query", "plan"]:
    cp = state.get("hitl_checkpoint_1")
    if cp and cp.status == HITLStatus.CORRECTED:
        return "user_query"
    return "plan"


# ── Build the main graph ──────────────────────────────────────

def build_graph():
    graph = StateGraph(TravelAgentState)

    graph.add_node("user_query",    handle_user_query_node)
    graph.add_node("out_of_scope",  out_of_scope_node)
    graph.add_node("hitl_1",        hitl_1_node)
    graph.add_node("plan",          plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("replan",        replan_node)
    graph.add_node("execute",       execute_node)
    graph.add_node("hitl_2",        hitl_2_node)
    graph.add_node("activities",    activities_node)

    graph.add_edge(START, "user_query")

    graph.add_conditional_edges(
        "user_query",
        route_user_query,
        {"out_of_scope": "out_of_scope", "hitl_1": "hitl_1", "end": END},
    )

    graph.add_edge("out_of_scope", END)

    graph.add_conditional_edges(
        "hitl_1",
        route_hitl_1,
        {"user_query": "user_query", "plan": "plan"},
    )

    graph.add_edge("plan", "validate_plan")

    graph.add_conditional_edges(
        "validate_plan",
        plan_valid,
        {"execute": "execute", "replan": "replan"},
    )

    graph.add_conditional_edges(
        "replan",
        replan_or_execute,
        {"plan": "plan", "execute": "execute"},
    )

    graph.add_edge("execute",    "hitl_2")
    graph.add_edge("hitl_2",     "activities")
    graph.add_edge("activities", END)

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