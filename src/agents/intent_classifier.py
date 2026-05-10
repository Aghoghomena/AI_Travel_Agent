import json
import os
from src.prompts.intent_classifier import build_intent_classifier_prompt
from src.state import IntentStatus, QueryState, Traveller
from src.utils.config import llm

class IntentClassifierAgent:
    """Agent responsible for classifying user intent and extracting structured information from the initial message.
    Returns a dict with status and extracted fields."""
    
    def classify_intent(self, user_message: str) -> dict:
        """Classifies the user's intent and extracts relevant information."""
        prompt = build_intent_classifier_prompt(user_message)
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback — treat as needs info if parse fails
            return {
                "status": IntentStatus.NEEDS_INFO,
                "missing_fields": ["travellers",
                                   "travel_month",
                                   "duration_nights"],
                "detected_fields": {},
                "parse_error": raw
            }

        return result
    
    def update_query_state(self, query_state: QueryState, classification: dict) -> QueryState:
        """
        Merges newly detected fields into existing QueryState.
        Called after a NEEDS_INFO or READY classification
        to preserve fields collected in earlier turns.
        """
        detected = classification.get("detected_fields", {})
        # Merge origins — don't overwrite existing travellers
        if detected.get("origins"):
            existing_iatas = {
                t.origin_city.lower()
                for t in query_state.travellers
            }
            for origin in detected["origins"]:
                if origin.lower() not in existing_iatas:
                    query_state.travellers.append(
                        Traveller(
                            name=f"Traveller {len(query_state.travellers) + 1}",
                            origin_city=origin
                        )
                    )

        if "travel_month" in detected and detected["travel_month"]:
            query_state.travel_month = detected["travel_month"]
        if detected.get("outbound_date") and not query_state.outbound_date:
            query_state.outbound_date = detected["outbound_date"]
        if "duration_nights" in detected and detected["duration_nights"]:
            query_state.duration_nights = detected["duration_nights"]
        if "region_preference" in detected and detected["region_preference"]:
            query_state.region_preferences = detected["region_preference"]
        return query_state
    

def run_intent_classifier(state:dict)-> dict:
    """
    LangGraph node function.
    Reads user_message from state, writes intent_status back.
    """
    agent = IntentClassifierAgent()
    user_message = state.get("user_message", "")

    result = agent.classify_intent(user_message)
    status = result.get("status", "NEEDS_INFO")

    # Map string to enum
    intent_map = {
        "OUT_OF_SCOPE": IntentStatus.OUT_OF_SCOPE,
        "NEEDS_INFO":   IntentStatus.NEEDS_INFO,
        "READY":        IntentStatus.READY,
    }
    intent_status = intent_map.get(status, IntentStatus.NEEDS_INFO)

    # Update query state with any detected fields
    updated_query_state = agent.update_query_state(
        result,
        state.get("query_state", QueryState())
    )

    updates = {
        "intent_status": intent_status,
        "query_state": updated_query_state,
    }

    # Add out of scope reason if applicable
    if intent_status == IntentStatus.OUT_OF_SCOPE:
        updates["out_of_scope_reason"] = result.get("reason", "")
        updates["agent_response"] = (
            f"That's outside what I can help with — "
            f"{result.get('reason', 'this system focuses on group travel planning')}.\n\n"
            f"I help groups of 1–5 people flying from different cities "
            f"find the cheapest destination to meet, including flights, "
            f"accommodation and things to do."
        )

    return updates


