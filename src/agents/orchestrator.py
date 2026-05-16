"""
Orchestrator Agent (ReWOO).
 
Implements Plan → Execute → Solve pattern.
 
PLAN:   LLM generates full XML execution plan before any API calls.
EXECUTE: Fires traveller agents (parallel), accommodation, currency.
SOLVE:  Runs ranker, returns ranked destinations.
 
LangGraph node: run_orchestrator(state) -> dict
"""

import xml.etree.ElementTree as ET
from typing import Literal
from langgraph.graph import StateGraph, START, END
 
from src.prompts.orchestrator import build_orchestrator_prompt
from src.agents.traveller import run_traveller_agent
from src.agents.accommodation import run_accommodation_agent
from src.agents.currency import run_currency_agent
from src.agents.ranker import run_ranker_agent
from src.memory.semantic import SemanticMemory
from src.memory.episodic import EpisodicMemory
from src.state import QueryState, DestinationResult, Traveller
from src.utils.config import llm
from src.state import OrchestratorState, query_state_from_dict, query_state_to_dict

# ── Tools ─────────────────────────────────────────────────────

def get_candidate_destinations(semantic_memory: SemanticMemory, region_preferences: str | None, prioritised_destinations: list[str], destination_locations=None, search_mode: str | None = None) -> list[str]:
    """
    Queries semantic memory for candidate destinations based on
    region preference. Reorders to put procedural winners first.
    Returns list of IATA codes.
    """
    # Specific mode — use exactly what the user asked for, skip semantic memory
    if search_mode == "specific" and destination_locations:
        return destination_locations[:6]

    # Map region_preferences to semantic query
    if semantic_memory is None:
        candidates = ["Turkey", "Portugal", "Netherlands", "United Kingdom", "Kenya", "Ghana"]
    else:
        region_lower = (region_preferences or "").lower().strip()

        if region_lower in ("africa",):
            query = "africa hub city meetup"
            region_filter = "africa"
        elif region_lower in ("europe",):
            query = "europe hub city meetup"
            region_filter = "europe"
        else:
            query = "hub city international meetup affordable"
            region_filter = None

        results = semantic_memory.query_destinations(
            query=query,
            region_filter=region_filter,
            n_results=8,
        )

        out_of_scope = {"DXB", "SIN"}
        candidates = [
            r["iata"] for r in results
            if r.get("iata") and r["iata"] not in out_of_scope
        ]

        if not candidates:
            candidates = ["Turkey", "Portugal", "Netherlands", "United Kingdom", "Kenya", "Ghana"]

    if prioritised_destinations:
        winners = [d for d in prioritised_destinations if d in candidates]
        rest = [d for d in candidates if d not in winners]
        candidates = winners + rest

    return candidates[:6]
    

def validate_plan(plan_xml: str, travellers: list, candidates: list[str]) -> tuple[bool, str]:
    """
    Validates the XML plan structure.
    Returns (is_valid, error_message).
    """
    try:
        root = ET.fromstring(plan_xml.strip())
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
 
    steps = root.findall("step")
    if not steps:
        return False, "Plan has no steps"
 
    tools = [s.get("tool") for s in steps]
 
    # Must have one traveller_agent step per traveller
    traveller_steps = [s for s in steps if s.get("tool") == "traveller_agent"]
    if len(traveller_steps) != len(travellers):
        return False, (
            f"Expected {len(travellers)} traveller_agent steps, "
            f"got {len(traveller_steps)}"
        )
 
    # Must have accommodation, currency, ranker
    for required in ("accommodation_agent", "currency_agent", "ranker_agent"):
        if required not in tools:
            return False, f"Missing required tool: {required}"
 
    # Currency must depend on traveller + accommodation steps
    currency_step = next(s for s in steps if s.get("tool") == "currency_agent")
    depends_on = currency_step.get("depends_on", "")
    if not depends_on:
        return False, "currency_agent step missing depends_on"
 
    return True, ""
 
  
def parse_plan_candidates(plan_xml: str) -> list[str]:
    """Extracts candidate destination IATAs from the plan."""
    try:
        root = ET.fromstring(plan_xml.strip())
        for step in root.findall("step"):
            if step.get("tool") == "accommodation_agent":
                dests = step.get("destinations", "")
                return [d.strip() for d in dests.split(",") if d.strip()]
    except ET.ParseError:
        pass
    return []
 

 # ── Nodes ─────────────────────────────────────────────────────
 
def plan_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 1: Queries semantic memory for candidates, then calls
    LLM to generate the XML execution plan.
    """
    query_state = query_state_from_dict(state.get("query_state", {}))
    # print(f"Planning Node with query_state: {query_state} in plan_node at orchestrator.py")  # Debug print
    semantic_memory = SemanticMemory()
    prioritised = state.get("prioritised_destinations", [])
    errors = list(state.get("errors", []))
 
    # Get candidate destinations
    candidates = get_candidate_destinations(
        semantic_memory=semantic_memory,
        region_preferences=query_state.region_preferences,
        prioritised_destinations=prioritised,
        destination_locations=query_state.destination_locations,
        search_mode=query_state.search_mode,
    )

    # Build and call LLM
    prompt = build_orchestrator_prompt(
        travellers=query_state.travellers,
        candidate_destinations=candidates,
        travel_month=query_state.travel_month,
        duration_nights=query_state.duration_nights,
        search_mode=query_state.search_mode or "anywhere",
    )
 
    response = llm.invoke(prompt)
    raw = response.content.strip()
    # print(f"LLM raw response for plan_node at orchestrator.py:\n{raw}\n")  # Debug print

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("xml"):
            raw = raw[3:]
    raw = raw.strip()
 
    return {
        **state,
        "rewoo_plan": raw,
        "candidate_destinations": candidates,
        "errors": errors,
    }
 

def validate_plan_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 2: Validates the XML plan structure.
    """
    plan_xml = state.get("rewoo_plan", "")
    query_state = query_state_from_dict(state.get("query_state", {})) 

    if isinstance(query_state, dict):
        travellers = query_state.get("travellers", [])
    else:
        travellers = query_state.travellers
    # print(f"Validating plan with travellers: {query_state} validate_plan_node at orchestrator.py")

    is_valid, error_msg = validate_plan(
        plan_xml=plan_xml,
        travellers=travellers,
        candidates=state.get("candidate_destinations", []),
    )

    # print(f"Plan validation result: is_valid={is_valid}, error_msg='{error_msg}'")  # Debug print

    return {
        **state,
        "rewoo_plan_valid": is_valid,
        "errors": state.get("errors", []) + ([error_msg] if not is_valid else []),
    }

def replan_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 3: Re-plans once on validation failure.
    On second failure falls back to a hardcoded minimal plan.
    """
    replan_count = state.get("replan_count", 0) + 1
 
    if replan_count >= 2:
        # Build fallback plan directly
        query_state = query_state_from_dict(state.get("query_state", {}))
        candidates = state.get("candidate_destinations", ["IST", "LIS", "AMS"])
        dest_str = ",".join(candidates)
 
        steps = []
        for i, t in enumerate(query_state.travellers, start=1):
            steps.append(
                f'  <step id="{i}" tool="traveller_agent" '
                f'traveller="{t.name}" origin="{t.origin_iata}" '
                f'destinations="{dest_str}" />'
            )
        n = len(query_state.travellers)
        steps.append(
            f'  <step id="{n+1}" tool="accommodation_agent" '
            f'destinations="{dest_str}" />'
        )
        traveller_ids = ",".join(str(i) for i in range(1, n + 2))
        steps.append(
            f'  <step id="{n+2}" tool="currency_agent" '
            f'depends_on="{traveller_ids}" />'
        )
        steps.append(
            f'  <step id="{n+3}" tool="ranker_agent" '
            f'depends_on="{n+2}" />'
        )
 
        fallback_plan = "<plan>\n" + "\n".join(steps) + "\n</plan>"
        return {
            **state,
            "replan_count": replan_count,
            "rewoo_plan": fallback_plan,
            "rewoo_plan_valid": True,
        }
 
    # Re-call plan_node logic
    return {**state, "replan_count": replan_count, "rewoo_plan_valid": False}

def execute_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 4: Executes the plan.
    Runs traveller agents, accommodation agent, currency agent.
    Traveller agents run sequentially here (parallel via Send() in graph.py Step 14).
    """
    query_state = query_state_from_dict(state.get("query_state", {}))
    candidates = state.get("candidate_destinations", [])
    # print(f"candidates {candidates}")
    errors = list(state.get("errors", []))
 
    # Build working state for sub-agents
    sub_state = {
        "candidate_destinations": candidates,
        "query_state": query_state,
        "episodic_memory": state.get("episodic_memory"),
        "semantic_memory": state.get("semantic_memory"),
    }
 
    # Run traveller agents (one per traveller)
    all_flight_results = []
    for traveller in query_state.travellers:
        result = run_traveller_agent(sub_state, traveller)
        all_flight_results.extend(result.get("flight_results", []))
        errors.extend(result.get("errors", []))
 
    # Group flights by destination — one DestinationResult per destination iata
    flights_by_dest: dict[str, list] = {}
    for f in all_flight_results:
        flights_by_dest.setdefault(f.destination_iata, []).append(f)

    required_origins = {t.origin_iata for t in query_state.travellers}
    destination_results = []

    for iata, dest_flights in flights_by_dest.items():
        # For region/anywhere mode, drop destinations not reachable by all travellers.
        # For specific mode, keep them — the user asked for these explicitly.
        if query_state.search_mode != "specific":
            covered_origins = {f.origin_iata for f in dest_flights}
            if not required_origins.issubset(covered_origins):
                missing = required_origins - covered_origins
                errors.append(f"{iata} dropped — no flights from: {missing}")
                continue

        dest = DestinationResult(city=iata, iata=iata)
        dest.flights = dest_flights
        dest.total_flight_cost_usd = sum(
            f.price_usd or f.price_local for f in dest_flights
        )
        destination_results.append(dest)

    # print(f"destination_results built from traveller output: {destination_results}")
 
    # Run accommodation agent
    accom_state = {**sub_state, "destination_results": destination_results}
    accom_result = run_accommodation_agent(accom_state)
    accommodation_results = accom_result.get("accommodation_results", [])
 
    # Attach accommodation to each destination
    accom_index = {a.destination_iata: a for a in accommodation_results}
    for dest in destination_results:
        accom = accom_index.get(dest.iata)
        if accom:
            dest.accommodation = accom
            dest.total_accommodation_usd = accom.total_usd
            dest.grand_total_usd = (
                (dest.total_flight_cost_usd or 0.0) +
                (dest.total_accommodation_usd or 0.0)
            )
 
    # Run currency agent
    currency_state = {
        **sub_state,
        "destination_results": destination_results,
        "flight_results": all_flight_results,
        "accommodation_results": accommodation_results,
    }
    currency_result = run_currency_agent(currency_state)
    destination_results = currency_result.get("destination_results", destination_results)
    exchange_rates = currency_result.get("exchange_rates", {})
 
    return {
        **state,
        "destination_results": destination_results,
        "flight_results": all_flight_results,
        "accommodation_results": accommodation_results,
        "exchange_rates": exchange_rates,
        "errors": errors,
    }

def solve_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 5: Runs the ranker agent and returns ranked destinations.
    """
    rank_result = run_ranker_agent({
        "destination_results": state.get("destination_results", []),
    })
 
    return {
        **state,
        "ranked_destinations": rank_result.get("ranked_destinations", []),
    }
 
 # ── Conditional edges ─────────────────────────────────────────
 
def plan_valid(state: OrchestratorState) -> Literal["execute", "replan"]:
    if state.get("rewoo_plan_valid"):
        return "execute"
    return "replan"
 
 
def replan_or_execute(state: OrchestratorState) -> Literal["plan", "execute"]:
    """After replan node — re-call plan or force execute with fallback."""
    if state.get("rewoo_plan_valid"):
        return "execute"
    if state.get("replan_count", 0) >= 2:
        return "execute"
    return "plan"


# ── Build graph ───────────────────────────────────────────────
 
def build_orchestrator_graph():
    graph = StateGraph(OrchestratorState)
 
    graph.add_node("plan",          plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("replan",        replan_node)
    graph.add_node("execute",       execute_node)
    graph.add_node("solve",         solve_node)
 
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        plan_valid,
        {"execute": "execute", "replan": "replan"},
    )
    graph.add_conditional_edges(
        "replan",
        replan_or_execute,
        {"plan": "plan", "execute": "execute"},
    )
    graph.add_edge("execute", "solve")
    graph.add_edge("solve",   END)
 
    return graph.compile()
 
 
orchestrator_agent = build_orchestrator_graph()
try:
    orchestrator_agent.get_graph().draw_mermaid_png(
        output_file_path="orchestrator_agent.png"
    )
    print("\nGraph saved as orchestrator_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")


# ── Entry point ───────────────────────────────────────────────
 
def run_orchestrator(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Runs full ReWOO plan → execute → solve cycle.
    """
    # print(f"Running orchestrator with initial state: {state}")  # Debug print
    result = orchestrator_agent.invoke({
        "query_state":               state.get("query_state", {}),
        "semantic_memory":           state.get("semantic_memory"),
        "episodic_memory":           state.get("episodic_memory"),
        "prioritised_destinations":  state.get("prioritised_destinations", []),
        "rewoo_plan":                None,
        "rewoo_plan_valid":          False,
        "replan_count":              0,
        "candidate_destinations":    [],
        "destination_results":       [],
        "flight_results":            [],
        "accommodation_results":     [],
        "exchange_rates":            {},
        "ranked_destinations":       [],
        "errors":                    [],
    })
 
    return {
        "query_state":             query_state_to_dict(query_state_from_dict(state.get("query_state", {}))),
        "candidate_destinations":  result.get("candidate_destinations", []),
        "destination_results":     result.get("destination_results", []),
        "ranked_destinations":     result.get("ranked_destinations", []),
        "flight_results":          result.get("flight_results", []),
        "accommodation_results":   result.get("accommodation_results", []),
        "exchange_rates":          result.get("exchange_rates", {}),
        "rewoo_plan":              result.get("rewoo_plan"),
        "errors":                  result.get("errors", []),
    }