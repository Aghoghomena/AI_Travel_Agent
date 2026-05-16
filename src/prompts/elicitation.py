import dspy
from typing import Literal, Optional
from pydantic import BaseModel, Field

from src.prompts.intent_classifier import TravellerField


# ── Pydantic model for structured output ─────────────────────────────────────

class ElicitedFields(BaseModel):
    """Fields newly extracted or corrected in this turn. None = not mentioned."""
    travellers: Optional[list[TravellerField]] = None
    travel_month: Optional[str] = None
    outbound_date: Optional[str] = None
    duration_nights: Optional[int] = None
    region_preferences: Optional[str] = None
    destination_locations: Optional[list[str]] = None
    search_mode: Optional[Literal["specific", "region", "anywhere"]] = None
    accommodation_needed: Optional[bool] = None
    max_budget_usd: Optional[float] = None
    direct_flights_only: Optional[bool] = None


# ── DSPy Signature ────────────────────────────────────────────────────────────

class Elicitation(dspy.Signature):
    """Collect missing travel details for a group travel planning system.

    The system helps 1-5 people flying from different cities find the cheapest
    destination to meet, covering flights, accommodation, and activities.

    Required fields (must all be present before search can run):
    1. travellers — at least one person with their origin city
    2. travel_month or outbound_date — when they want to travel
    3. duration_nights — how many nights (convert "one week" → 7, etc.)

    Optional fields to extract if mentioned:
    - region_preferences — preferred region or "anywhere"
    - destination_locations — specific cities/countries they want to visit
    - search_mode — "specific" (exact destinations named), "region" (broad area),
      or "anywhere" (no preference)
    - accommodation_needed — false if user says they'll handle it themselves
    - max_budget_usd — max per-person budget in USD if stated
    - direct_flights_only — true if user explicitly wants no layovers

    Turn strategy:
    - First turn (nothing collected): ask for everything at once in a single
      friendly question so the user can answer in one go.
    - Follow-up turns: ask ONLY for what is still missing. Never ask for
      something already in current_state. Extract every field the user provides
      even if you only asked for one.
    - Once all required fields are present: set question to empty string.

    Never mention IATA codes, field names, or technical internals.
    Keep responses short and friendly.
    """

    user_message: str = dspy.InputField(
        desc="The user's latest message"
    )
    current_state: str = dspy.InputField(
        desc="Summary of what has already been collected. Never ask for these again."
    )
    conversation_history: str = dspy.InputField(
        desc="Prior turns formatted as 'User: ...' / 'Assistant: ...'. Empty on first turn."
    )

    question: str = dspy.OutputField(
        desc=(
            "The follow-up question to ask the user. "
            "Empty string when all required fields are present."
        )
    )
    updated_fields: ElicitedFields = dspy.OutputField(
        desc=(
            "Fields newly extracted or corrected this turn. "
            "Omit (leave None) any field not mentioned — do not repeat already-collected values."
        )
    )


# ── Few-shot examples (for DSPy optimizers) ──────────────────────────────────

ELICITATION_EXAMPLES: list[dspy.Example] = [
    dspy.Example(
        user_message="Two of us — Dublin and Toronto, travelling in June for 3 nights",
        current_state="travellers=[], travel_month=None, duration_nights=None",
        conversation_history="",
        question="Any particular part of the world, or happy for anywhere?",
        updated_fields=ElicitedFields(
            travellers=[
                TravellerField(name="Traveller 1", origin_city="Dublin"),
                TravellerField(name="Traveller 2", origin_city="Toronto"),
            ],
            travel_month="June",
            duration_nights=3,
        ),
    ).with_inputs("user_message", "current_state", "conversation_history"),

    dspy.Example(
        user_message="Three of us — Lagos, Paris and London, July, one week, budget $400 each, direct flights only",
        current_state="travellers=[], travel_month=None, duration_nights=None",
        conversation_history="",
        question="Any particular part of the world, or happy for anywhere?",
        updated_fields=ElicitedFields(
            travellers=[
                TravellerField(name="Traveller 1", origin_city="Lagos"),
                TravellerField(name="Traveller 2", origin_city="Paris"),
                TravellerField(name="Traveller 3", origin_city="London"),
            ],
            travel_month="July",
            duration_nights=7,
            max_budget_usd=400.0,
            direct_flights_only=True,
        ),
    ).with_inputs("user_message", "current_state", "conversation_history"),

    dspy.Example(
        user_message="anywhere is fine, and we're staying with friends so no need for hotels",
        current_state="travellers=['Lagos', 'Dublin'], travel_month='August', duration_nights=5",
        conversation_history="Assistant: Any particular part of the world?\nUser: anywhere is fine, and we're staying with friends so no need for hotels",
        question="",
        updated_fields=ElicitedFields(
            region_preferences="anywhere",
            search_mode="anywhere",
            accommodation_needed=False,
        ),
    ).with_inputs("user_message", "current_state", "conversation_history"),

    dspy.Example(
        user_message="we want to go to Lisbon or Rome",
        current_state="travellers=['Dublin', 'Toronto'], travel_month='July', duration_nights=5",
        conversation_history="Assistant: Any particular part of the world?\nUser: we want to go to Lisbon or Rome",
        question="",
        updated_fields=ElicitedFields(
            destination_locations=["Lisbon", "Rome"],
            search_mode="specific",
        ),
    ).with_inputs("user_message", "current_state", "conversation_history"),
]
