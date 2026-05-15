import json
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from src.prompts.intent_classifier import build_intent_classifier_prompt
from src.prompts.elicitation import build_elicitation_prompt
from src.state import IntentStatus, QueryState, Traveller, query_state_from_dict, query_state_to_dict
from src.utils.config import llm
from data.iata import resolve_iata

load_dotenv()
MAX_ELICITATION_TURNS = int(os.getenv("MAX_ELICITATION_TURNS", 5))


class IntakeState(dict):
    user_message: str
    query_state: QueryState
    conversation_history: list[dict]
    turn_count: int
    intent_status: IntentStatus | None
    elicitation_question: str | None
    elicitation_complete: bool
    turn_limit_reached: bool
    agent_response: str | None


def _all_required_present(query_state: QueryState) -> bool:
    return query_state.is_ready_for_search()

def _parse(raw: str) -> dict | None:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def _merge(query_state: QueryState, fields: dict) -> QueryState:
    print(f"Merging the travellers: {query_state.travellers} on _merge")
    existing = {t.origin_city.lower() for t in query_state.travellers}
    for traveller in fields.get("travellers", []):
        origin = traveller.get("origin_city", "")
        if origin and origin.lower() not in existing:
            iata = resolve_iata(origin)
            if not iata:
                print(f"Warning: could not resolve IATA for '{origin}', skipping")
                continue
            query_state.travellers.append(Traveller(
                name=traveller.get("name", f"Traveller {len(query_state.travellers) + 1}"),
                origin_city=origin,
                origin_iata=resolve_iata(origin),
            ))
            existing.add(origin.lower())
    if fields.get("travel_month") and not query_state.travel_month:
        query_state.travel_month = fields["travel_month"]
    if fields.get("outbound_date") and not query_state.outbound_date:
        query_state.outbound_date = fields["outbound_date"]
    if fields.get("duration_nights") and not query_state.duration_nights:
        query_state.duration_nights = fields["duration_nights"]
    if fields.get("region_preferences") and not query_state.region_preferences:
        query_state.region_preferences = fields["region_preferences"]
    print(f"Merged query state: {query_state} with new fields: {fields} on _merge")
    return query_state



# ── Nodes ─────────────────────────────────────────────────────

def classify_node(state: IntakeState) -> IntakeState:
    raw = llm.invoke(build_intent_classifier_prompt(state["user_message"], state.get("query_state")) ).content.strip()

    result = _parse(raw) or {"status": "NEEDS_INFO", "detected_fields": {}}
    status = result.get("status", "NEEDS_INFO")
    query_state = _merge(query_state_from_dict(state.get("query_state", {})), result.get("detected_fields", {}))

    agent_response = None
    if status == "OUT_OF_SCOPE":
        reason = result.get("reason", "")
        agent_response = (
            f"That's outside what I can help with — {reason}.\n\n"
            f"I help groups of 1–5 people flying from different cities "
            f"find the cheapest destination to meet."
        )
    print(f"Classify node result: status={status}, query_state={query_state}, agent_response={agent_response} on classify_node with raw={raw}")

    return {
        **state,
        "query_state": query_state_to_dict(query_state),
        "intent_status": IntentStatus(status.lower()),
        "elicitation_complete": status == "READY",   # ← add this
        "agent_response": agent_response,
    }


def elicit_node(state: IntakeState) -> IntakeState:
    query_state = query_state_from_dict(state.get("query_state", {}))
    history = state.get("conversation_history", [])
    turn_count = state.get("turn_count", 0) + 1

    raw = llm.invoke(build_elicitation_prompt(state["user_message"], query_state, history)).content.strip()

    result = _parse(raw)
    print(f"Raw elicit node output: {raw} on elicit_node")
    if result:
        query_state = _merge(query_state, result.get("updated_fields", {}))
        complete = _all_required_present(query_state)
        question = "" if complete else result.get("question", "")
    else:
        missing = query_state.missing_info()
        question = f"Could you tell me your {missing[0]}?" if missing else "Could you confirm your details?"
        complete = False

    new_history = list(history)
    if state.get("user_message"):
        new_history.append({"role": "user", "content": state["user_message"]})
    if question:
        new_history.append({"role": "assistant", "content": question})

    print(f"Elicit node result: query_state={query_state}, question={question}, complete={complete}, turn_count={turn_count} on elicit_node with raw={raw}")

    return {
        **state,
        "query_state": query_state_to_dict(query_state),
        "elicitation_question": question,
        "elicitation_complete": complete,
        "intent_status": IntentStatus.READY if complete else state.get("intent_status"),
        "turn_count": turn_count,
        "turn_limit_reached": turn_count >= MAX_ELICITATION_TURNS and not complete,
        "conversation_history": new_history,
    }


# ── Routing ───────────────────────────────────────────────────

def route_after_classify(state: IntakeState) -> Literal["elicit", "end"]:
    print(f"Routing after classify with intent_status={state.get('intent_status')} on route_after_classify")
    if state.get("intent_status") == IntentStatus.NEEDS_INFO:
        return "elicit"
    return "end"


# ── Graph ─────────────────────────────────────────────────────

def build_intake_graph():
    graph = StateGraph(IntakeState)
    graph.add_node("classify", classify_node)
    graph.add_node("elicit",   elicit_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify,
                                {"elicit": "elicit", "end": END})
    graph.add_edge("elicit", END)
    return graph.compile()


intake_agent = build_intake_graph()

try:
    intake_agent.get_graph().draw_mermaid_png(output_file_path="user_query_agent.png")
    print("\nGraph saved as user_query_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")


def handle_user_query(state: dict) -> dict:
    result = intake_agent.invoke({
        "user_message":        state.get("user_message", ""),
        "query_state":         state.get("query_state", QueryState()),
        "conversation_history": state.get("conversation_history", []),
        "turn_count":          state.get("turn_count", 0),
        "intent_status":       state.get("intent_status"),  # was: None
        "elicitation_question": None,
        "elicitation_complete": False,
        "turn_limit_reached":  False,
        "agent_response":      None,
    })
    return {
        "intent_status":        result.get("intent_status"),
        "query_state":          result.get("query_state"),
        "agent_response":       result.get("agent_response"),
        "elicitation_question": result.get("elicitation_question"),
        "elicitation_complete": result.get("elicitation_complete", False),
        "turn_limit_reached":   result.get("turn_limit_reached", False),
        "conversation_history": result.get("conversation_history", []),
        "turn_count":           result.get("turn_count", 0),
    }
