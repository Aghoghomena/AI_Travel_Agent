import os
from typing import Literal
from dotenv import load_dotenv
import dspy
from langgraph.graph import StateGraph, START, END
from src.prompts.intent_classifier import IntentClassification, IntentDetectedFields
from src.prompts.elicitation import Elicitation, ElicitedFields
from src.state import IntentStatus, QueryState, Traveller, query_state_from_dict, query_state_to_dict
from src.utils.config import llm  # also triggers dspy.configure()
from data.iata import resolve_iata

load_dotenv()
MAX_ELICITATION_TURNS = int(os.getenv("MAX_ELICITATION_TURNS", 3))


# ── DSPy classifier module ────────────────────────────────────────────────────

class IntentClassifier(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(IntentClassification)

    def forward(self, user_message: str, collected_info: str = "") -> dspy.Prediction:
        return self.predict(user_message=user_message, collected_info=collected_info)


_classifier = IntentClassifier()


class Elicitor(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(Elicitation)

    def forward(self, user_message: str, current_state: str, conversation_history: str) -> dspy.Prediction:
        return self.predict(
            user_message=user_message,
            current_state=current_state,
            conversation_history=conversation_history,
        )


_elicitor = Elicitor()


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


def _build_state_summary(query_state: QueryState) -> str:
    origins = [t.origin_city for t in query_state.travellers]
    return (
        f"travellers={origins or []}, "
        f"travel_month={query_state.travel_month or 'None'}, "
        f"duration_nights={query_state.duration_nights or 'None'}, "
        f"region_preferences={query_state.region_preferences or 'None'}, "
        f"accommodation_needed={query_state.accommodation_needed}, "
        f"max_budget_usd={query_state.max_budget_usd or 'None'}, "
        f"direct_flights_only={query_state.direct_flights_only}"
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "Assistant" if turn["role"] == "assistant" else "User"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _build_collected_info(query_state: QueryState) -> str:
    origins = [t.origin_city for t in query_state.travellers]
    month = query_state.travel_month
    nights = query_state.duration_nights
    if not any([origins, month, nights]):
        return ""
    return (
        f"Already collected: origins={origins or 'none'}, "
        f"travel_month={month or 'none'}, "
        f"duration_nights={nights or 'none'}"
    )


def _merge(query_state: QueryState, fields: dict) -> QueryState:
 
    existing = {t.origin_city.lower() for t in query_state.travellers}
    for traveller in fields.get("travellers", []):
        origin = traveller.get("origin_city", "")
        if origin and origin.lower() not in existing:
            iata = resolve_iata(origin)
            if not iata:
                print(f"Warning: could not resolve IATA for '{origin}', skipping")
                continue
            query_state.travellers.append(Traveller(
                name=traveller.get("name") or f"Traveller {len(query_state.travellers) + 1}",
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
    if fields.get("search_mode") and not query_state.search_mode:
        query_state.search_mode = fields["search_mode"]
    if fields.get("destination_locations") and not query_state.destination_locations:
        query_state.destination_locations = fields["destination_locations"]
    if "accommodation_needed" in fields:
        query_state.accommodation_needed = fields["accommodation_needed"]
    if fields.get("max_budget_usd") and not query_state.max_budget_usd:
        query_state.max_budget_usd = fields["max_budget_usd"]
    if "direct_flights_only" in fields:
        query_state.direct_flights_only = fields["direct_flights_only"]
    return query_state



# ── Nodes ─────────────────────────────────────────────────────

def classify_node(state: IntakeState) -> IntakeState:
    query_state = query_state_from_dict(state.get("query_state", {}))
    collected_info = _build_collected_info(query_state)

    try:
        prediction = _classifier(
            user_message=state["user_message"],
            collected_info=collected_info,
        )
        status = prediction.status
        detected: IntentDetectedFields = prediction.detected_fields
        reason = prediction.reason or ""
    except Exception as exc:
        print(f"[classify_node] DSPy error: {exc}")
        status = "NEEDS_INFO"
        detected = IntentDetectedFields()
        reason = ""

    # Convert Pydantic model to the dict shape _merge expects
    fields_dict = detected.model_dump()
    fields_dict["travellers"] = [t.model_dump() for t in (detected.travellers or [])]
    query_state = _merge(query_state, fields_dict)

    agent_response = None
    if status == "OUT_OF_SCOPE":
        agent_response = (
            f"That's outside what I can help with — {reason}.\n\n"
            f"I help groups of 1–5 people flying from different cities "
            f"find the cheapest destination to meet."
        )

    return {
        **state,
        "query_state": query_state_to_dict(query_state),
        "intent_status": IntentStatus(status.lower()),
        "elicitation_complete": status == "READY",
        "agent_response": agent_response,
    }


def elicit_node(state: IntakeState) -> IntakeState:
    query_state = query_state_from_dict(state.get("query_state", {}))
    history = state.get("conversation_history", [])
    turn_count = state.get("turn_count", 0) + 1

    try:
        prediction = _elicitor(
            user_message=state["user_message"],
            current_state=_build_state_summary(query_state),
            conversation_history=_format_history(history),
        )
        updated: ElicitedFields = prediction.updated_fields
        question = prediction.question or ""

        # exclude_none=True so only fields mentioned this turn reach _merge
        fields_dict = updated.model_dump(exclude_none=True)
        query_state = _merge(query_state, fields_dict)
    except Exception as exc:
        print(f"[elicit_node] DSPy error: {exc}")
        missing = query_state.missing_info()
        question = f"Could you tell me your {missing[0]}?" if missing else "Could you confirm your details?"

    complete = _all_required_present(query_state)
    if complete:
        question = ""

    new_history = list(history)
    if state.get("user_message"):
        new_history.append({"role": "user", "content": state["user_message"]})
    if question:
        new_history.append({"role": "assistant", "content": question})

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

    if (
        state.get("intent_status") == IntentStatus.NEEDS_INFO
        and not state.get("turn_limit_reached")
        and not state.get("elicitation_complete")
    ):
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
