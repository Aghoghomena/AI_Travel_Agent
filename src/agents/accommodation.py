"""
Accommodation Agent (mocked).
 
Returns realistic nightly hotel costs per destination based on
2025/2026 average 3-star hotel rates from Hotels.com Price Index
and independent travel price surveys.
 
Prices are in USD. No external API call — fully mocked.
Checks episodic memory cache first. Writes results back to cache.
 
LangGraph node: run_accommodation_agent(state) -> dict
"""
 
from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.state import AccommodationResult, AccommodationState, QueryState
from data.iata import MOCK_HOTEL_PRICES


DEFAULT_PRICE = 120.0   # fallback for unknown destinations

# ── Tools ─────────────────────────────────────────────────────
 
def get_mock_price(destination_iata: str) -> float:
    """Returns the mock nightly hotel price for a destination."""
    return MOCK_HOTEL_PRICES.get(destination_iata.upper(), DEFAULT_PRICE)


def check_episodic_cache(episodic_memory, destination_iata: str, nights: int) -> AccommodationResult | None:
    """Checks episodic memory for a cached accommodation result. No-op until Step 11."""
    if episodic_memory is None:
        return None
    try:
        return episodic_memory.get_accommodation(destination_iata, nights)
    except Exception:
        return None
 

def write_episodic_cache(episodic_memory, result: AccommodationResult) -> None:
    """Writes an accommodation result to episodic memory. No-op until Step 11."""
    if episodic_memory is None:
        return
    try:
        episodic_memory.set_accommodation(result)
    except Exception:
        pass


# ── Nodes ─────────────────────────────────────────────────────
 
def fetch_accommodation_node(state: AccommodationState) -> AccommodationState:
    """
    Node 1: Returns mocked accommodation costs for each candidate
    destination. Checks episodic cache first. if new write the results into episodic memory
    """
    destinations = state.get("candidate_destinations", [])
    nights = state.get("duration_nights", 1)
    episodic_memory = state.get("episodic_memory")
 
    results = list(state.get("accommodation_results", []))
 
    for dest_iata in destinations:
        # Check cache first
        cached = check_episodic_cache(episodic_memory, dest_iata, nights)
        if cached:
            results.append(AccommodationResult(
                destination_city=cached.destination_city,
                destination_iata=cached.destination_iata,
                price_per_night_local=cached.price_per_night_local,
                currency=cached.currency,
                nights=cached.nights,
                total_usd=cached.total_usd,
                fetched_at=cached.fetched_at,
                from_cache=True,
            ))
            continue
 
        price_per_night = get_mock_price(dest_iata)
        total_usd = round(price_per_night * nights, 2)
 
        result = AccommodationResult(
            destination_city=dest_iata,    # city name resolved at display time
            destination_iata=dest_iata,
            price_per_night_local=price_per_night,
            currency="USD",
            nights=nights,
            total_usd=total_usd,
            fetched_at=datetime.utcnow().isoformat(),
            from_cache=False,
        )
 
        write_episodic_cache(episodic_memory, result)
        results.append(result)
 
    return {**state, "accommodation_results": results}

# ── Build the agent graph ─────────────────────────────────────
 
def build_accommodation_graph():
    graph = StateGraph(AccommodationState)
 
    graph.add_node("fetch_accommodation", fetch_accommodation_node)
 
    graph.add_edge(START, "fetch_accommodation")
    graph.add_edge("fetch_accommodation", END)
 
    return graph.compile()

accommodation_agent = build_accommodation_graph()
try:
    accommodation_agent.get_graph().draw_mermaid_png(output_file_path="accommodation_agent.png")
    print("\nGraph saved as accommodation_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")

# ── Entry point for main travel agent graph ───────────────────
 
def run_accommodation_agent(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Runs accommodation lookup for all candidate destinations.
    """
    query_state = state.get("query_state", QueryState())
 
    result = accommodation_agent.invoke({
        "candidate_destinations": state.get("candidate_destinations", []),
        "duration_nights": query_state.duration_nights or 1,
        "accommodation_results": [],
        "episodic_memory": state.get("episodic_memory"),
    })
 
    return {
        "accommodation_results": result.get("accommodation_results", []),
    }