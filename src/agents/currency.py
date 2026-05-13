"""
Currency Agent.
 
Converts USD totals to each traveller's local currency for display.
All internal calculations stay in USD — conversion is display-only.
 
For each destination × traveller combination:
  - Calculates that traveller's share: flight_usd + accommodation_share_usd
  - Converts to traveller's local currency
  - Stores in per_traveller_breakdown on DestinationResult
 
Uses frankfurter.app for ECB currencies (EUR, GBP, TRY, ZAR, MAD...).
Uses static fallback rates for African currencies not in ECB
(NGN, GHS, KES, EGP, XOF, ETB, TZS, RWF, UGX, DZD, TND).
 
LangGraph node: run_currency_agent(state) -> dict
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.tools.frankfurter import (fetch_rates,convert_usd_to_local,get_currency_for_iata,)
from src.state import QueryState, CurrencyState, TravellerCost, Traveller


# ── Nodes ─────────────────────────────────────────────────────
 
def fetch_rates_node(state: CurrencyState) -> CurrencyState:
    """
    Node 1: Resolves each traveller's currency from their origin IATA
    and fetches all needed USD → local rates in one API call.
    """
    travellers = state.get("travellers", [])
 
    # Resolve currency for each traveller
    updated_travellers = []
    currencies_needed = set()
 
    for t in travellers:
        currency = t.currency if t.currency else get_currency_for_iata(t.origin_iata)
        currencies_needed.add(currency)
        # Return updated traveller with currency set
        updated_travellers.append(Traveller(name=t.name,
            origin_city=t.origin_city,
            origin_iata=t.origin_iata,
            currency=currency,
            preferences=t.preferences,
        ))
 
    rates = fetch_rates(list(currencies_needed))
 
    return {
        **state,
        "travellers": updated_travellers,
        "exchange_rates": rates,
    }

def convert_costs_node(state: CurrencyState) -> CurrencyState:
    """
    Node 2: For each destination, calculates per-traveller cost
    breakdown in both USD and local currency.
 
    Per-traveller total = their flight USD + equal share of accommodation USD.
    """
    travellers = state.get("travellers", [])
    destination_results = state.get("destination_results", [])
    flight_results = state.get("flight_results", [])
    accommodation_results = state.get("accommodation_results", [])
    rates = state.get("exchange_rates", {})
    errors = list(state.get("errors", []))
 
    # Index flights by (origin_iata, destination_iata)
    flight_index: dict[tuple, float] = {}
    for f in flight_results:
        key = (f.origin_iata, f.destination_iata)
        # Keep cheapest if duplicates
        if key not in flight_index or f.price_usd < flight_index[key]:
            flight_index[key] = f.price_usd or f.price_local
 
    # Index accommodation by destination_iata
    accom_index: dict[str, float] = {}
    for a in accommodation_results:
        accom_index[a.destination_iata] = a.total_usd or 0.0
 
    updated_results = []
 
    for dest in destination_results:
        dest_iata = dest.iata
        accom_total_usd = accom_index.get(dest_iata, 0.0)
        accom_share_usd = round(accom_total_usd / len(travellers), 2) if travellers else 0.0
 
        breakdown = []
        total_flight_usd = 0.0
 
        for t in travellers:
            flight_usd = flight_index.get((t.origin_iata, dest_iata), 0.0)
            total_usd = round(flight_usd + accom_share_usd, 2)
            rate = rates.get(t.currency, 1.0)
            total_local = convert_usd_to_local(total_usd, t.currency, rates)
            total_flight_usd += flight_usd
 
            breakdown.append(TravellerCost(
                traveller_name=t.name,
                origin_iata=t.origin_iata,
                currency=t.currency,
                flight_usd=flight_usd,
                accommodation_share_usd=accom_share_usd,
                total_usd=total_usd,
                total_local=total_local,
                exchange_rate=rate,
            ))
 
        # Update destination totals
        dest.total_flight_cost_usd = round(total_flight_usd, 2)
        dest.total_accommodation_usd = accom_total_usd
        dest.grand_total_usd = round(total_flight_usd + accom_total_usd, 2)
        dest.per_traveller_breakdown = breakdown
 
        updated_results.append(dest)
 
    return {
        **state,
        "updated_destination_results": updated_results,
        "errors": errors,
    }
 

# ── Build the agent graph ─────────────────────────────────────
 
def build_currency_graph():
    graph = StateGraph(CurrencyState)
 
    graph.add_node("fetch_rates", fetch_rates_node)
    graph.add_node("convert_costs", convert_costs_node)
 
    graph.add_edge(START, "fetch_rates")
    graph.add_edge("fetch_rates", "convert_costs")
    graph.add_edge("convert_costs", END)
 
    return graph.compile()
 
 
currency_agent = build_currency_graph()
try:
    currency_agent.get_graph().draw_mermaid_png(output_file_path="currency_agent.png")
    print("\nGraph saved as currency_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")
 

# ── Entry point for main travel agent graph ───────────────────
 
def run_currency_agent(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Converts USD totals to each traveller's local currency.
    """
    result = currency_agent.invoke({
        "travellers": state.get("query_state", QueryState()).travellers,
        "destination_results": state.get("destination_results", []),
        "flight_results": state.get("flight_results", []),
        "accommodation_results": state.get("accommodation_results", []),
        "exchange_rates": {},
        "updated_destination_results": [],
        "errors": [],
    })
 
    return {
        "destination_results": result.get("updated_destination_results", []),
        "exchange_rates": result.get("exchange_rates", {}),
        "travellers": result.get("travellers", []),
        "errors": result.get("errors", []),
    }
 