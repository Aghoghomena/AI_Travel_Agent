"""
Traveller Agent.
 
One instance per origin city (1-5 travellers, dynamic fan-out).
Checks episodic memory cache first. On cache miss calls Amadeus.
Resolves city names to IATA codes — static map first, LLM fallback.
 
LangGraph node: run_traveller_agent(state, origin_iata) -> dict
"""
 
import json
from typing import Literal
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from src.prompts.traveller import build_iata_resolution_prompt
from src.tools.amadeus import search_flights, resolve_iata
from src.state import FlightResult, QueryState, Traveller
from src.utils.config import llm
from src.state import TravellerState

# ── Tools ─────────────────────────────────────────────────────
 
def resolve_iata_llm(city_name: str) -> str | None:
    """
    LLM fallback for IATA resolution when city is not in static map.Returns IATA code string or None if unresolvable.
    """
    prompt = build_iata_resolution_prompt(city_name)
    response = llm.invoke(prompt)
    raw = response.content.strip()
 
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
 
    try:
        parsed = json.loads(raw)
        if parsed.get("confidence") == "high" and parsed.get("iata_code"):
            return parsed["iata_code"]
    except json.JSONDecodeError:
        pass
 
    return None

def get_or_resolve_iata(traveller: Traveller) -> tuple[str | None, str | None]:
    """
    Returns (iata_code, error).Tries static map first, then LLM fallback.
    """
    # Already resolved
    if traveller.origin_iata:
        return traveller.origin_iata, None
 
    # Static map
    iata = resolve_iata(traveller.origin_city)
    if iata:
        return iata, None
 
    # LLM fallback
    iata = resolve_iata_llm(traveller.origin_city)
    if iata:
        return iata, None
 
    return None, (
        f"Could not resolve IATA code for '{traveller.origin_city}'. "
        f"Please provide a major city name or airport code."
    )

def check_episodic_cache(episodic_memory, origin_iata: str, destination_iata: str, date: str) -> FlightResult | None:
    """Checks episodic memory for a cached flight leg result. Returns FlightResult if cache hit, None otherwise.
    Episodic memory is wired in Step 11 — safe no-op until then.
    """
    if episodic_memory is None:
        return None
    try:
        return episodic_memory.get_flight_leg(origin_iata, destination_iata, date)
    except Exception:
        return None
 
 
def write_episodic_cache(episodic_memory, result: FlightResult) -> None:
    """Writes a flight result to episodic memory cache."""
    if episodic_memory is None:
        return
    try:
        episodic_memory.set_flight_leg(result)
    except Exception:
        pass

# ── Nodes ─────────────────────────────────────────────────────

def resolve_iata_node(state: TravellerState) -> TravellerState:
    """
    Node 1: Resolves the traveller's origin city to an IATA code.
    Static map first, LLM fallback if not found.
    Updates traveller.origin_iata on state.
    """
    traveller = state["traveller"]
    iata, error = get_or_resolve_iata(traveller)
 
    if error:
        return {**state, "iata_resolution_error": error}
 
    # Update traveller with resolved IATA
    updated_traveller = Traveller(
        name=traveller.name,
        origin_city=traveller.origin_city,
        origin_iata=iata,
        preferences=traveller.preferences,
    )
 
    return {
        **state,
        "traveller": updated_traveller,
        "iata_resolution_error": None,
    }

def search_flights_node(state: TravellerState) -> TravellerState:
    """
    Node 2: Searches flights from this traveller's origin to each
    candidate destination. Checks episodic cache first.
    Writes new results back to cache.
    """
    if state.get("iata_resolution_error"):
        return state
 
    traveller = state["traveller"]
    origin_iata = traveller.origin_iata
    destinations = state.get("candidate_destinations", [])
    outbound_date = state.get("outbound_date", "")
    episodic_memory = state.get("episodic_memory")
 
    flight_results = list(state.get("flight_results", []))
    search_errors = list(state.get("search_errors", []))
 
    for dest_iata in destinations:
        if dest_iata == origin_iata:
            continue
 
        # Check cache first
        cached = check_episodic_cache(episodic_memory, origin_iata, dest_iata, outbound_date)
        if cached:
            cached_result = FlightResult(
                origin_iata=cached.origin_iata,
                destination_iata=cached.destination_iata,
                price_local=cached.price_local,
                currency=cached.currency,
                price_usd=cached.price_usd,
                fetched_at=cached.fetched_at,
                from_cache=True,
            )
            flight_results.append(cached_result)
            continue
 
        # Cache miss — call Amadeus
        results = search_flights(
            origin_iata=origin_iata,
            destination_iata=dest_iata,
            travel_month=state.get("travel_month"),
            outbound_date=outbound_date if outbound_date else None,
        )
 
        if results:
            for r in results:
                write_episodic_cache(episodic_memory, r)
            flight_results.extend(results)
        else:
            search_errors.append(
                f"No flights found: {origin_iata} → {dest_iata}"
            )
 
    return {
        **state,
        "flight_results": flight_results,
        "search_errors": search_errors,
    }


# ── Conditional edge functions ────────────────────────────────
 
def iata_resolved(state: TravellerState) -> Literal["search", "end"]:
    """Skip flight search if IATA resolution failed."""
    if state.get("iata_resolution_error"):
        return "end"
    return "search"

# ── Build the agent graph ─────────────────────────────────────
 
def build_traveller_graph():
    graph = StateGraph(TravellerState)
 
    graph.add_node("resolve_iata", resolve_iata_node)
    graph.add_node("search_flights", search_flights_node)
 
    graph.add_edge(START, "resolve_iata")
    graph.add_conditional_edges(
        "resolve_iata",
        iata_resolved,
        {"search": "search_flights", "end": END},
    )
    graph.add_edge("search_flights", END)
 
    return graph.compile()

traveller_agent = build_traveller_graph()
try:
    traveller_agent.get_graph().draw_mermaid_png(output_file_path="traveller_agent.png")
    print("\nGraph saved as traveller_agent.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")

def run_traveller_agent(state: dict, traveller: Traveller) -> dict:
    """
    Entry point for the main travel agent graph.
    Called once per traveller in parallel fan-out (Step 14).
 
    Args:
        state: Main TravelAgentState dict.
        traveller: The specific Traveller this instance handles.
 
    Returns:
        Partial state dict with flight_results and any errors.
    """
    result = traveller_agent.invoke({
        "traveller": traveller,
        "candidate_destinations": state.get("candidate_destinations", []),
        "outbound_date": state.get("query_state", QueryState()).outbound_date or "",
        "travel_month": state.get("query_state", QueryState()).travel_month,
        "flight_results": [],
        "iata_resolution_error": None,
        "search_errors": [],
        "episodic_memory": state.get("episodic_memory"),
    })
 
    errors = []
    if result.get("iata_resolution_error"):
        errors.append(result["iata_resolution_error"])
    errors.extend(result.get("search_errors", []))
 
    return {
        "flight_results": result.get("flight_results", []),
        "errors": errors,
    }
 

