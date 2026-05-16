from __future__ import annotations
from typing import Annotated, Any
from dataclasses import dataclass, field
from enum import Enum 
import operator
from typing import TypedDict



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
    currency: str | None = None
    preferences: dict[str, Any] = field(default_factory=dict)

@dataclass
class Outbound_flights:
    city: str
    country: str
    code: str


@dataclass
class FlightResult:
    """Flight price for one origin → destination leg."""
    origin_iata: str
    destination_iata: str
    price_local: float
    currency: str
    avg_cost_per_night: float
    outbound_flights: dict[str, Any] = field(default_factory=dict)
    price_usd: float | None = None
    fetched_at: str | None = None
    from_cache: bool = False


@dataclass
class AccommodationResult:
    """Accommodation cost for one destination."""
    destination_city: str
    destination_iata: str
    price_per_night_local: float
    price_per_night_usd: float
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
    per_traveller_breakdown: list | None = None


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
    destination_locations: list[str] = field(default_factory=list)
    search_mode: str | None = None  # "specific" | "region" | "anywhere"
    # in state.py
    search_mode: str | None = None  # "specific" | "region" | "anywhere"
    max_budget_usd: float | None = None
    accommodation_needed: bool = True
    direct_flights_only: bool = False


    def is_ready_for_search(self) -> bool:
        """Determines if we have enough information to run a search."""
        return (
            len(self.travellers) > 0 and
            (self.travel_month is not None or self.outbound_date is not None) and
            self.duration_nights is not None and
            self.search_mode is not None
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
class TravelAgentState2:
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



class TravelAgentState(TypedDict, total=False):
    session_id: str
    conversation_history: list[dict]
    user_message: str
    agent_response: str | None
    intent_status: str | None
    out_of_scope_reason: str | None
    query_state: dict
    elicitation_question: str | None
    elicitation_complete: bool
    turn_count: int
    turn_limit_reached: bool
    search_status: str | None
    flight_results: list
    accommodation_results: list
    exchange_rates: dict
    destination_results: list
    ranked_destinations: list
    activity_destinations: list
    activities: dict
    activities_fetched: bool
    episodic_cache_hit: bool
    cached_result_date: str | None
    memory_write_complete: bool
    hitl_checkpoint_1: dict | None
    hitl_checkpoint_2: dict | None
    hitl_plan_checkpoint: dict | None
    hitl_plan_feedback: str | None
    hitl_1_confirmed: bool
    hitl_plan_approved: bool
    awaiting_hitl: bool
    hitl_replan_complete: bool
    rewoo_plan: str | None
    rewoo_plan_valid: bool
    current_step: str | None
    past_searches_summary: str | None
    hitl_plan_approved: bool | None = None
    hitl_plan_feedback: str | None = None
    hitl_replan_complete: bool = False
    errors: list


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
        "hitl_plan_checkpoint": None,
        "hitl_plan_feedback": None,
        "hitl_plan_approved": False,
        "awaiting_hitl": False,
        "hitl_replan_complete": False,

        # ReWOO
        "rewoo_plan": None,
        "rewoo_plan_valid": False,
        "current_step": None,

        # Errors
        "errors": [],
    }


# ── Internal state for the elicitation agent's graph ────────────────────
class ElicitationState(dict):
    user_message: str
    raw_llm_output: str | None
    parsed_output: dict | None
    retry_count: int
    turn_count: int
    error: str | None
    query_state: QueryState
    elicitation_question: str | None
    elicitation_complete: bool
    turn_limit_reached: bool
    corrections: list[str]
    conversation_history: list[dict]


 
# ── Internal state for this agent's graph ────────────────────
class TravellerState(dict):
    traveller: Traveller
    candidate_destinations: list[str]       # IATA codes
    outbound_date: str
    flight_results: list[FlightResult]
    iata_resolution_error: str | None
    search_errors: list[str]
    travel_month: str | None                # e.g. 'June' — used if no exact date
    episodic_memory: object | None          # EpisodicMemory instance (Step 11)
 

 # ── Internal state for this agent's graph ────────────────────
class AccommodationState(dict):
    candidate_destinations: list[str]       # IATA codes
    duration_nights: int
    accommodation_results: list[AccommodationResult]
    episodic_memory: object | None


# ── Per-traveller cost breakdown ──────────────────────────────
@dataclass
class TravellerCost:
    """Cost breakdown for one traveller at one destination."""
    traveller_name: str
    origin_iata: str
    currency: str 
    flight_usd: float
    accommodation_share_usd: float
    total_usd: float
    total_local: float
    exchange_rate: float


# ── Internal state for this agent's graph ────────────────────
class ActivitiesState(dict):
    destination_iatas: list[str]
    activities: dict[str, list[Activity]]   # iata → activities
    semantic_memory: object | None          # SemanticMemory (Step 11)
    errors: list[str]


# ── Internal state for this agent's graph ────────────────────
class RankerState(dict):
    destination_results: list
    raw_llm_output: str | None
    parsed_ranking: list[dict] | None
    retry_count: int
    error: str | None
    ranked_destinations: list

# ── Internal states for memory agent ───────────────────────────────────────────

class MemoryReadState(dict):
    query_state: QueryState
    session_id: str
    cache_hit: bool
    cached_result: dict | None
    prioritised_destinations: list[str]
    past_searches_summary: str | None
    errors: list[str]
 
 
class MemoryWriteState(dict):
    query_state: QueryState
    session_id: str
    ranked_destinations: list
    flight_results: list
    accommodation_results: list
    exchange_rates: dict
    hitl_checkpoint_1: object | None
    hitl_checkpoint_2: object | None
    errors: list[str]



# ── Internal state ────────────────────────────────────────────
class OrchestratorState(dict):
    query_state: QueryState
    semantic_memory: object
    episodic_memory: object
    prioritised_destinations: list[str]
    rewoo_plan: str | None
    rewoo_plan_valid: bool
    replan_count: int
    candidate_destinations: list[str]      # IATA codes
    destination_results: list[DestinationResult]
    flight_results: list
    accommodation_results: list
    exchange_rates: dict
    ranked_destinations: list
    errors: list[str]
    hitl_plan_feedback: str | None
    hitl_plan_approved: bool | None
    hitl_replan_complete: bool

# ── Internal state for this agent's graph ────────────────────
class CurrencyState(dict):
    travellers: list[Traveller]
    destination_results: list           # list of DestinationResult
    flight_results: list                # list of FlightResult
    accommodation_results: list         # list of AccommodationResult
    exchange_rates: dict[str, float]    # currency → rate from USD
    updated_destination_results: list
    errors: list[str]
 
 
class IntakeState(dict):
    user_message: str
    query_state: QueryState
    conversation_history: list[dict]
    turn_count: int

    # classifier
    raw_classify_output: str | None
    classification: dict | None
    classify_retry_count: int
    classify_error: str | None
    intent_status: IntentStatus | None
    out_of_scope_reason: str | None
    agent_response: str | None

    # elicitation
    raw_elicit_output: str | None
    parsed_elicit_output: dict | None
    elicit_retry_count: int
    elicit_error: str | None
    elicitation_question: str | None
    elicitation_complete: bool
    corrections: list
    turn_limit_reached: bool


# ─────────────────────────────────────────
# QueryState ↔ dict helpers
# (keeps QueryState as plain dict in graph state for MemorySaver compatibility)
# ─────────────────────────────────────────
 
def query_state_to_dict(qs: QueryState) -> dict:
    """Converts a QueryState dataclass to a plain serialisable dict."""
    return {
        "travellers": [
            {
                "name": t.name,
                "origin_city": t.origin_city,
                "origin_iata": t.origin_iata,
                "currency": getattr(t, "currency", ""),
                "preferences": getattr(t, "preferences", {}),
            }
            for t in qs.travellers
        ],
        "travel_month":     qs.travel_month,
        "outbound_date":    qs.outbound_date,
        "return_date":      qs.return_date,
        "duration_nights":  qs.duration_nights,
        "region_preferences": qs.region_preferences,
        "destination_locations": qs.destination_locations,
        "search_mode": qs.search_mode,
        "candidate_destinations": [],  # DestinationResult not serialised here
        "max_budget_usd": qs.max_budget_usd,
        "accommodation_needed": qs.accommodation_needed,
        "direct_flights_only": qs.direct_flights_only,
    }
 
 
def _make_traveller(t: dict) -> "Traveller":
    """Creates a Traveller from dict, handling optional currency field."""
    kwargs = {
        "name": t.get("name", ""),
        "origin_city": t.get("origin_city", ""),
        "origin_iata": t.get("origin_iata", ""),
    }
    # currency is optional — only pass if Traveller accepts it
    import inspect
    sig = inspect.signature(Traveller.__init__)
    if "currency" in sig.parameters:
        kwargs["currency"] = t.get("currency", "")
    if "preferences" in sig.parameters:
        kwargs["preferences"] = t.get("preferences", {})
    return Traveller(**kwargs)
 
 
def query_state_from_dict(d: dict) -> QueryState:
    """Converts a plain dict back to a QueryState dataclass."""
    if isinstance(d, QueryState):
        return d  # already a QueryState — no-op
    if not isinstance(d, dict):
        return QueryState()
 
    travellers = [
        _make_traveller(t)
        for t in d.get("travellers", [])
    ]
 
    qs = QueryState()
    qs.travellers          = travellers
    qs.travel_month        = d.get("travel_month")
    qs.outbound_date       = d.get("outbound_date")
    qs.return_date         = d.get("return_date")
    qs.duration_nights     = d.get("duration_nights")
    qs.region_preferences  = d.get("region_preferences")
    qs.destination_locations = d.get("destination_locations") or []
    qs.search_mode         = d.get("search_mode")
    qs.max_budget_usd      = d.get("max_budget_usd")
    qs.accommodation_needed = d.get("accommodation_needed", True)
    qs.direct_flights_only  = d.get("direct_flights_only", False)
    return qs
 