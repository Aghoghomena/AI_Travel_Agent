import json
import os
from typing import Literal
from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END
from src.prompts.intent_classifier import INTENT_CLASSIFIER_PROMPT
from src.state import IntentStatus, QueryState, Traveller
from src.utils.config import llm
from langchain_core.messages import SystemMessage, HumanMessage
from data.iata import IATA_MAP, resolve_iata



# ── Internal state for this agent's graph ────────────────────
class ClassifierState(dict):
    user_message: str
    raw_llm_output: str | None
    classification: dict | None
    retry_count: int
    error: str | None
    query_state: QueryState
    intent_status: IntentStatus | None
    out_of_scope_reason: str | None
    agent_response: str | None


# ── Tools ─────────────────────────────────────────────────────


def validate_classification(classification: dict) -> tuple[bool, str]:
    """
    Validates the LLM output is well-formed.
    Returns (is_valid, error_message)
    """
    if not isinstance(classification, dict):
        return False, "Output is not a dict"

    status = classification.get("status")
    if status not in ["OUT_OF_SCOPE", "NEEDS_INFO", "READY"]:
        return False, f"Invalid status: {status}"

    if status == "OUT_OF_SCOPE":
        if not classification.get("reason"):
            return False, "OUT_OF_SCOPE missing reason"

    if status == "NEEDS_INFO":
        if "missing_fields" not in classification:
            return False, "NEEDS_INFO missing missing_fields"
        if "detected_fields" not in classification:
            return False, "NEEDS_INFO missing detected_fields"

    if status == "READY":
        detected = classification.get("detected_fields", {})
        if not detected.get("origins"):
            return False, "READY but no origins detected"
        if not detected.get("travel_month") and \
           not detected.get("outbound_date"):
            return False, "READY but no travel time detected"
        if not detected.get("duration_nights"):
            return False, "READY but no duration detected"

    return True, ""


# ── Nodes ─────────────────────────────────────────────────────

def classify_node(state: ClassifierState) -> ClassifierState:
    """
    Node 1: Calls LLM to classify the user message.
    Writes raw output to state.
    """
    response = llm.invoke(INTENT_CLASSIFIER_PROMPT)

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return {**state, "raw_llm_output": raw}


def validate_node(state: ClassifierState) -> ClassifierState:
    """
    Node 2: Validates the raw LLM output.
    Parses JSON and checks structure.
    Writes classification or error to state.
    """
    raw = state.get("raw_llm_output", "")

    try:
        classification = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            **state,
            "classification": None,
            "error": f"JSON parse failed: {e}"
        }

    is_valid, error_msg = validate_classification(classification)

    if not is_valid:
        return {
            **state,
            "classification": None,
            "error": error_msg
        }

    return {
        **state,
        "classification": classification,
        "error": None
    }



def retry_node(state: ClassifierState) -> ClassifierState:
    """
    Node 3: Increments retry counter.
    LangGraph routes back to classify_node if under limit.
    On max retries falls back to NEEDS_INFO.
    """
    retry_count = state.get("retry_count", 0) + 1

    if retry_count >= 2:
        # Fallback — treat as needs info rather than failing
        fallback = {
            "status": "NEEDS_INFO",
            "missing_fields": [
                "travellers",
                "travel_month",
                "duration_nights"
            ],
            "detected_fields": {
                "origins": [],
                "travel_month": None,
                "duration_nights": None,
                "region_preferences": None
            }
        }
        return {
            **state,
            "retry_count": retry_count,
            "classification": fallback,
            "error": None
        }

    return {**state, "retry_count": retry_count}

def update_state_node(state: ClassifierState) -> ClassifierState:
    """
    Node 4: Merges classification results into query_state.
    Resolves city names to IATA codes where possible.
    Sets intent_status and agent_response on state.
    """
    classification = state["classification"]
    status = classification["status"]
    detected = classification.get("detected_fields", {})

    intent_map = {
        "OUT_OF_SCOPE": IntentStatus.OUT_OF_SCOPE,
        "NEEDS_INFO":   IntentStatus.NEEDS_INFO,
        "READY":        IntentStatus.READY,
    }
    intent_status = intent_map[status]

    # Merge detected origins into query_state
    query_state = state.get("query_state", QueryState())
    existing_cities = {
        t.origin_city.lower()
        for t in query_state.travellers
    }

    for origin in detected.get("origins", []):
        if origin.lower() not in existing_cities:
            iata = resolve_iata(origin)
            query_state.travellers.append(
                Traveller(
                    name=f"Traveller {len(query_state.travellers)+1}",
                    origin_city=origin,
                    origin_iata=iata
                )
            )
            existing_cities.add(origin.lower())

    # Merge other fields
    if detected.get("travel_month") and not query_state.travel_month:
        query_state.travel_month = detected["travel_month"]

    if detected.get("outbound_date") and not query_state.outbound_date:
        query_state.outbound_date = detected["outbound_date"]

    if detected.get("duration_nights") and \
       not query_state.duration_nights:
        query_state.duration_nights = detected["duration_nights"]

    if detected.get("region_preferences") and \
       not query_state.region_preferences:
        query_state.region_preferences = detected["region_preferences"]

    # Build agent response for out of scope
    agent_response = None
    out_of_scope_reason = None

    if intent_status == IntentStatus.OUT_OF_SCOPE:
        out_of_scope_reason = classification.get("reason", "")
        agent_response = (
            f"That's outside what I can help with — "
            f"{out_of_scope_reason}.\n\n"
            f"I help groups of 1–5 people flying from different "
            f"cities find the cheapest destination to meet, "
            f"including flights, accommodation and things to do."
        )

    return {
        **state,
        "query_state": query_state,
        "intent_status": intent_status,
        "out_of_scope_reason": out_of_scope_reason,
        "agent_response": agent_response,
    }


# ── Conditional edge functions ────────────────────────────────

def should_retry(state: ClassifierState) -> Literal["retry", "update"]:
    """Routes to retry if validation failed, update if valid."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    return "update"

def retry_or_end(state: ClassifierState) -> Literal["classify", "update"]:
    """After retry node — go back to classify or force update."""
    if state.get("classification"):
        return "update"
    return "classify"


# ── Build the agent graph ─────────────────────────────────────
def build_intent_classifier_graph():
    graph = StateGraph(ClassifierState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("update_state", update_state_node)

    # Add edges
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "validate")
    # validate → retry or update_state
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "retry",
            "update": "update_state"
        }
    )
    # retry → classify again or force update
    graph.add_conditional_edges(
        "retry",
        retry_or_end,
        {
            "classify": "classify",
            "update": "update_state"
        }
    )
     # update_state → END always
    graph.add_edge("update_state", END)
    return graph.compile()

# Instantiate the graph once at module level so it can be reused across calls

intent_classifier_agent = build_intent_classifier_graph()
try:
    intent_classifier_agent.get_graph().draw_mermaid_png(output_file_path="intent_classifier_agent.png")
    print("\nGraph saved as intent_classifier_agent.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")


def run_intent_classifier(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Runs the full intent classifier sub-graph.
    """
    result = intent_classifier_agent.invoke({
        "user_message": state["user_message"],
        "raw_llm_output": None,
        "classification": None,
        "retry_count": 0,
        "error": None,
        "query_state": state.get("query_state", QueryState()),
        "intent_status": None,
        "out_of_scope_reason": None,
        "agent_response": None,
    })

    return {
        "intent_status": result["intent_status"],
        "query_state": result["query_state"],
        "out_of_scope_reason": result.get("out_of_scope_reason"),
        "agent_response": result.get("agent_response"),
    }