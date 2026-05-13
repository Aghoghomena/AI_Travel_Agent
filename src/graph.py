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
 
from src.agents.intent_classifier import run_intent_classifier
from src.agents.elicitation_manager import run_elicitation_manager
from src.agents.orchestrator import run_orchestrator
from src.agents.memory_agent import run_memory_read, run_memory_write
from src.agents.activities import run_activities_agent
from src.state import (
    IntentStatus, HITLCheckpoint,
    HITLStatus, QueryState
)


# ── Node functions ────────────────────────────────────────────
 
def intent_classifier_node(state: dict) -> dict:
    return run_intent_classifier(state)
 
 
def elicitation_node(state: dict) -> dict:
    return run_elicitation_manager(state)
 
 
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
    Interrupt node — confirms collected travel details before
    any API calls fire.
    """
    query_state = state.get("query_state", QueryState())
 
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
 
    checkpoint = HITLCheckpoint(
        checkpoint_id="hitl_1",
        message=message,
        status=HITLStatus.PENDING,
    )
 
    user_response = interrupt(message)
    checkpoint.user_response = user_response
 
    if user_response.strip().lower() in ("yes", "y", "correct", "yep", "yeah"):
        checkpoint.status = HITLStatus.CONFIRMED
        return {
            "hitl_checkpoint_1": checkpoint,
            "awaiting_hitl": False,
        }
    else:
        # Correction — loop back to elicitation
        checkpoint.status = HITLStatus.CORRECTED
        checkpoint.correction = {"raw": user_response}
        return {
            "hitl_checkpoint_1": checkpoint,
            "awaiting_hitl": False,
            "user_message": user_response,
            "elicitation_complete": False,
        }
 
 
# ── HITL Checkpoint 2 — post-ranking destination selection ────
 
def hitl_2_node(state: dict) -> dict:
    """
    Interrupt node — shows ranked destinations and asks which
    ones to fetch activities for.
    """
    ranked = state.get("ranked_destinations", [])
 
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
    }

# ── Conditional routing ───────────────────────────────────────
 
def route_intent(state: dict) -> Literal["out_of_scope", "elicitation", "hitl_1"]:
    status = state.get("intent_status")
    if status == IntentStatus.OUT_OF_SCOPE:
        return "out_of_scope"
    if status == IntentStatus.NEEDS_INFO:
        return "elicitation"
    return "hitl_1"
 
 
def route_hitl_1(state: dict) -> Literal["elicitation", "memory_read"]:
    cp = state.get("hitl_checkpoint_1")
    if cp and cp.status == HITLStatus.CORRECTED:
        return "elicitation"
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
    query_state = state.get("query_state", QueryState())
    return [
        Send("traveller_node", {**state, "current_traveller": t})
        for t in query_state.travellers
    ]
 
 
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
    query_state = state.get("query_state", QueryState())
    return [
        Send("traveller_node", {**state, "current_traveller": t})
        for t in query_state.travellers
    ]