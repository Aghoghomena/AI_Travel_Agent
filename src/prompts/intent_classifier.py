import dspy
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Pydantic models for structured output ────────────────────────────────────

class TravellerField(BaseModel):
    name: Optional[str] = None
    origin_city: str


class IntentDetectedFields(BaseModel):
    travellers: list[TravellerField] = Field(default_factory=list)
    travel_month: Optional[str] = None
    outbound_date: Optional[str] = None
    duration_nights: Optional[int] = None
    region_preferences: Optional[str] = None
    destination_locations: Optional[list[str]] = None
    search_mode: Literal["specific", "region", "anywhere"] = "anywhere"
    accommodation_needed: bool = True
    max_budget_usd: Optional[float] = None
    direct_flights_only: bool = False


# ── DSPy Signature ────────────────────────────────────────────────────────────

class IntentClassification(dspy.Signature):
    """Classify a user message for a group travel planning system.

    The system helps 1-5 people flying from different cities find the cheapest
    destination to meet, covering flights, accommodation, and activities.

    Classification rules:
    - OUT_OF_SCOPE: nothing to do with group travel planning.
    - NEEDS_INFO: travel-related but missing at least one required field
      (at least one origin city, travel month or dates, duration in nights).
    - READY: has ALL required fields.

    search_mode rules:
    - "specific": user named exact destination cities or countries to visit.
    - "region": user named a continent or broad region (e.g. "Europe", "Africa").
    - "anywhere": no destination preference expressed.
    """

    user_message: str = dspy.InputField(
        desc="The user's raw message to classify"
    )
    collected_info: str = dspy.InputField(
        desc=(
            "Summary of fields already collected from previous turns. "
            "Classify based on what is STILL missing across ALL collected info, "
            "not just this message alone. Empty string on the first turn."
        )
    )

    status: Literal["OUT_OF_SCOPE", "NEEDS_INFO", "READY"] = dspy.OutputField(
        desc="Intent classification result"
    )
    reason: str = dspy.OutputField(
        desc="Brief reason why out of scope. Empty string if status is not OUT_OF_SCOPE."
    )
    detected_fields: IntentDetectedFields = dspy.OutputField(
        desc="Travel fields extracted from the user message."
    )


# ── Few-shot examples (available for DSPy optimizers) ────────────────────────

INTENT_EXAMPLES: list[dspy.Example] = [
    dspy.Example(
        user_message="Write me a CV",
        collected_info="",
        status="OUT_OF_SCOPE",
        reason="CV writing is not travel planning",
        detected_fields=IntentDetectedFields(),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="Can you book flights for us?",
        collected_info="",
        status="OUT_OF_SCOPE",
        reason="This system finds and ranks destinations but does not book flights",
        detected_fields=IntentDetectedFields(),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="Me and two mates want to meet up somewhere cheap",
        collected_info="",
        status="NEEDS_INFO",
        reason="",
        detected_fields=IntentDetectedFields(search_mode="anywhere"),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="Find somewhere warm in Europe for 4 of us in July for a week",
        collected_info="",
        status="NEEDS_INFO",
        reason="",
        detected_fields=IntentDetectedFields(
            travel_month="July",
            duration_nights=7,
            region_preferences="Europe",
            search_mode="region",
        ),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="I'm in Dublin, friend in Lagos, June, 3 nights",
        collected_info="",
        status="READY",
        reason="",
        detected_fields=IntentDetectedFields(
            travellers=[
                TravellerField(name="Traveller 1", origin_city="Dublin"),
                TravellerField(name="Traveller 2", origin_city="Lagos"),
            ],
            travel_month="June",
            duration_nights=3,
            search_mode="anywhere",
        ),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="Dublin, Toronto, Lagos — we want to go to Lisbon or Rome, mid July, 5 nights",
        collected_info="",
        status="READY",
        reason="",
        detected_fields=IntentDetectedFields(
            travellers=[
                TravellerField(name="Traveller 1", origin_city="Dublin"),
                TravellerField(name="Traveller 2", origin_city="Toronto"),
                TravellerField(name="Traveller 3", origin_city="Lagos"),
            ],
            travel_month="July",
            duration_nights=5,
            destination_locations=["Lisbon", "Rome"],
            search_mode="specific",
        ),
    ).with_inputs("user_message", "collected_info"),

    dspy.Example(
        user_message="3 nights",
        collected_info="Already collected: origins=['Dublin', 'Lagos'], travel_month='June', duration_nights=none",
        status="READY",
        reason="",
        detected_fields=IntentDetectedFields(
            travellers=[
                TravellerField(name="Traveller 1", origin_city="Dublin"),
                TravellerField(name="Traveller 2", origin_city="Lagos"),
            ],
            travel_month="June",
            duration_nights=3,
            search_mode="anywhere",
        ),
    ).with_inputs("user_message", "collected_info"),
]
