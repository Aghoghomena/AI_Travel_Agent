from __future__ import annotations
from typing import Annotated, Any
from dataclasses import dataclass, field
from enum import Enum
import operator



# ─────────────────────────────────────────
# Enums
# ─────────────────────────────────────────

class IntentStatus(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_INFO   = "needs_info"
    READY        = "ready"

class SearchStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETE   = "complete"
    FAILED     = "failed"

class HITLStatus(str, Enum):
    PENDING   = "pending"
    CONFIRMED = "confirmed"
    REJECTED  = "rejected"
    CORRECTED = "corrected"

# ─────────────────────────────────────────
# Data Classes IATA stands for International Air Transport Association
# ─────────────────────────────────────────

@dataclass
class Traveller:
    """Represents a traveller in the group."""
    name: str
    origin_city: str
    origin_iata: str
    preferences: dict[str, Any] = field(default_factory=dict)

@dataclass
class FlightResult:
    """Flight price for one origin → destination leg."""
    origin_iata: str
    destination_iata: str
    price_local: float
    currency: str
    price_usd: float | None = None
    fetched_at: str | None = None
    from_cache: bool = False

@dataclass
class AccommodationResult:
    """Accommodation cost for one destination."""
    destination_city: str
    destination_iata: str
    price_per_night_local: float
    currency: str
    nights: int
    total_usd: float | None = None
    fetched_at: str | None = None
    from_cache: bool = False

@dataclass
class Activity:
    """One thing to do at a destination."""
    name: str
    category: str
    estimated_cost_usd: float | None = None
    distance_km: float | None = None
    rating: float | None = None
    address: str | None = None
    is_free: bool = False

@dataclass
class DestinationResult:
    """Fully aggregated result for one candidate destination."""
    city: str
    iata: str
    flights: list[FlightResult] = field(default_factory=list)
    accommodation: AccommodationResult | None = None
    activities: list[Activity] = field(default_factory=list)
    total_flight_cost_usd: float | None = None
    total_accommodation_usd: float | None = None
    grand_total_usd: float | None = None
    rank: int | None = None


@dataclass
class HITLCheckpoint:
    """State of a human-in-the-loop confirmation."""
    checkpoint_id: str
    message: str
    status: HITLStatus = HITLStatus.PENDING
    user_response: str | None = None
    correction: dict[str, Any] | None = None

# ─────────────────────────────────────────
# Query state — built up during elicitation
# ─────────────────────────────────────────

@dataclass
class QueryState:
    """Everything needed to run a search. Filled progressively during elicitation."""
    travellers: list[Traveller] = field(default_factory=list)
    travel_month: str | None = None
    outbound_date: str | None = None # resolved from month
    return_date: str | None = None # resolved from duration
    candidate_destinations: list[DestinationResult] = field(default_factory=list)
    duration_nights: int | None = None
    region_preferences: str | None = None # Europe, Anywhere, etc.
    candidate_destinations: list[DestinationResult] = field(default_factory=list)

    def is_ready_for_search(self) -> bool:
        """Determines if we have enough information to run a search."""
        return (
            len(self.travellers) > 0 and
            self.travel_month is not None and
            self.duration_nights is not None
        )
    
    def missing_info(self) -> list[str]:
        """Returns a list of missing information needed to run a search."""
        missing = []
        if len(self.travellers) == 0:
            missing.append("travellers")
        if self.travel_month is None:
            missing.append("travel month")
        if self.duration_nights is None:
            missing.append("trip duration")
        return missing
    


# ─────────────────────────────────────────
# Main graph state
# ─────────────────────────────────────────
class TravelAgentState:
    """Encapsulates the entire state of the travel agent, including the current query and any HITL checkpoints."""
    session_id: str
    conversation_history:  Annotated[list[dict], operator.add]
    user_messages: str
    agent_responses: str | None

    # ── Intent ────────────────────────────────────────────────
    intent_status: IntentStatus | None
    out_of_scope_reason: str | None

    # ── Elicitation ───────────────────────────────────────────
    query_state: QueryState
    elicitation_question: str | None
    elicitation_complete: bool

    # ── Search ───────────────────────────────────────────────
    search_status: SearchStatus | None
    flightResults:  Annotated[list[FlightResult], operator.add]
    accommodationResults:  list[AccommodationResult]
    exchange_rate: dict[str, float]          # currency → USD rate
    destinationResults:  list[DestinationResult]
    ranked_destinations: list[DestinationResult]

    # ── Activities ───────────────────────────────────────────────
    activities_fetched: bool
    activities: dict[str, list[Activity]]     # destination_iata → activities   

    # ── Memory ────────────────────────────────────────────────
    episodic_cache_hit: bool
    cached_result_date: str | None
    memory_write_complete: bool
    
    # ── Human-in-the-loop checkpoints ───────────────────────────────
    hitl_search_confirmation: HITLCheckpoint | None
    hitl_destination_confirmation: HITLCheckpoint | None
    awaiting_hitl_response: bool

    # ── ReWOO plan ────────────────────────────────────────────
    rewoo_plan: str | None                    # raw XML plan from orchestrator
    rewoo_plan_valid: bool
    current_step: str | None

    # ── Errors ────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


# ─────────────────────────────────────────
# Default state factory
# ─────────────────────────────────────────
def create_initial_state(
    session_id: str,
    user_message: str
) -> dict:
    """Creates a fresh state dict for a new conversation turn."""
    return {
        # Conversation
        "session_id": session_id,
        "conversation_history": [],
        "user_message": user_message,
        "agent_response": None,

        # Intent
        "intent_status": None,
        "out_of_scope_reason": None,

        # Elicitation
        "query_state": QueryState(),
        "elicitation_question": None,
        "elicitation_complete": False,

        # Search
        "search_status": SearchStatus.PENDING,
        "flight_results": [],
        "accommodation_results": [],
        "exchange_rates": {},
        "destination_results": [],
        "ranked_destinations": [],

        # Activities
        "activity_destinations": [],
        "activities_fetched": False,

        # Memory
        "episodic_cache_hit": False,
        "cached_result_date": None,
        "memory_write_complete": False,

        # HITL
        "hitl_checkpoint_1": None,
        "hitl_checkpoint_2": None,
        "awaiting_hitl": False,

        # ReWOO
        "rewoo_plan": None,
        "rewoo_plan_valid": False,
        "current_step": None,

        # Errors
        "errors": [],
    }
