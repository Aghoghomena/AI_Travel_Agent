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
 
from src.prompts.orchestrator import build_orchestrator_prompt
from src.agents.traveller import run_traveller_agent
from src.agents.accommodation import run_accommodation_agent
from src.agents.currency import run_currency_agent
from src.agents.ranker import run_ranker_agent
from src.memory.semantic import SemanticMemory
from src.state import QueryState, DestinationResult, Traveller
from src.utils.config import llm
from src.state import OrchestratorState, query_state_from_dict

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

    # Must start with memory_read
    for required in ("memory_read", "accommodation_agent", "currency_agent", "ranker_agent", "memory_write"):
        if required not in tools:
            return False, f"Missing required tool: {required}"

    # Must have one traveller_agent step per traveller
    traveller_steps = [s for s in steps if s.get("tool") == "traveller_agent"]
    if len(traveller_steps) != len(travellers):
        return False, (
            f"Expected {len(travellers)} traveller_agent steps, "
            f"got {len(traveller_steps)}"
        )

    # Currency must have depends_on
    currency_step = next(s for s in steps if s.get("tool") == "currency_agent")
    if not currency_step.get("depends_on", ""):
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
 
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("xml"):
                raw = raw[3:]
        raw = raw.strip()
    except Exception as exc:
        errors.append(f"plan_node LLM error: {exc}")
        raw = ""  # empty string → validate_plan will fail → replan fallback

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

        steps = ['  <step id="1" tool="memory_read" />']
        for i, t in enumerate(query_state.travellers, start=2):
            steps.append(
                f'  <step id="{i}" tool="traveller_agent" '
                f'traveller="{t.name}" origin="{t.origin_iata}" '
                f'destinations="{dest_str}" depends_on="1" />'
            )
        n = len(query_state.travellers)
        accom_id = n + 2
        steps.append(
            f'  <step id="{accom_id}" tool="accommodation_agent" '
            f'destinations="{dest_str}" depends_on="1" />'
        )
        traveller_ids = ",".join(str(i) for i in range(2, n + 2))
        currency_id = accom_id + 1
        steps.append(
            f'  <step id="{currency_id}" tool="currency_agent" '
            f'depends_on="{traveller_ids},{accom_id}" />'
        )
        ranker_id = currency_id + 1
        steps.append(f'  <step id="{ranker_id}" tool="ranker_agent" depends_on="{currency_id}" />')
        write_id = ranker_id + 1
        steps.append(f'  <step id="{write_id}" tool="memory_write" depends_on="{ranker_id}" />')

        fallback_plan = "<plan>\n" + "\n".join(steps) + "\n</plan>"
        return {
            **state,
            "replan_count": replan_count,
            "rewoo_plan": fallback_plan,
            "rewoo_plan_valid": True,
        }
 
    # Re-call plan_node logic
    return {**state, "replan_count": replan_count, "rewoo_plan_valid": False}

def _parse_steps(plan_xml: str) -> dict[str, ET.Element]:
    """Returns {step_id: element} from the plan XML."""
    root = ET.fromstring(plan_xml.strip())
    return {s.get("id"): s for s in root.findall("step")}


def _execution_order(steps: dict[str, ET.Element]) -> list[list[str]]:
    """
    Topological sort respecting depends_on.
    Returns groups of step IDs that can run in parallel.
    """
    groups: list[list[str]] = []
    completed: set[str] = set()
    remaining = set(steps.keys())

    while remaining:
        ready = [
            sid for sid in remaining
            if all(
                d.strip() in completed
                for d in steps[sid].get("depends_on", "").split(",")
                if d.strip()
            )
        ]
        if not ready:
            raise RuntimeError(f"Plan deadlock — unresolvable dependencies in steps: {remaining}")
        groups.append(ready)
        completed.update(ready)
        remaining -= set(ready)

    return groups


def execute_node(state: dict) -> dict:
    """
    Parses rewoo_plan XML and executes every step in dependency order.
    Handles all tools: memory_read, traveller_agent, accommodation_agent,
    currency_agent, ranker_agent, memory_write.
    Search steps are skipped automatically on a memory cache hit.
    """
    from src.agents.memory_agent import run_memory_read, run_memory_write

    query_state = query_state_from_dict(state.get("query_state", {}))
    errors = list(state.get("errors", []))

    try:
        steps = _parse_steps(state["rewoo_plan"])
        order = _execution_order(steps)
    except (ET.ParseError, RuntimeError) as exc:
        errors.append(f"Plan execution aborted: {exc}")
        return {**state, "errors": errors}

    base_state = {
        "candidate_destinations": state.get("candidate_destinations", []),
        "query_state": query_state,
        "episodic_memory": state.get("episodic_memory"),
        "semantic_memory": state.get("semantic_memory"),
    }

    cache_hit = False
    all_flight_results: list = []
    accommodation_results: list = []
    destination_results: list = []
    exchange_rates: dict = {}
    ranked_destinations: list = []

    SEARCH_TOOLS = {"traveller_agent", "accommodation_agent", "currency_agent", "ranker_agent", "memory_write"}

    for group in order:
        for sid in group:
            step = steps[sid]
            tool = step.get("tool")

            try:
                if tool == "memory_read":
                    result = run_memory_read({**state, **base_state})
                    state = {**state, **result}
                    cache_hit = state.get("episodic_cache_hit", False)
                    if cache_hit:
                        ranked_destinations = state.get("ranked_destinations", [])
                    errors.extend(result.get("errors", []))

                elif cache_hit and tool in SEARCH_TOOLS:
                    continue  # cached results already in state

                elif tool == "traveller_agent":
                    traveller_name = step.get("traveller")
                    destinations = [d.strip() for d in step.get("destinations", "").split(",") if d.strip()]
                    traveller = next((t for t in query_state.travellers if t.name == traveller_name), None)
                    if traveller is None:
                        errors.append(f"Plan step {sid}: traveller '{traveller_name}' not found")
                        continue
                    result = run_traveller_agent(
                        {**base_state, "candidate_destinations": destinations or base_state["candidate_destinations"]},
                        traveller,
                    )
                    all_flight_results.extend(result.get("flight_results", []))
                    errors.extend(result.get("errors", []))

                elif tool == "accommodation_agent":
                    destinations = [d.strip() for d in step.get("destinations", "").split(",") if d.strip()]
                    result = run_accommodation_agent(
                        {**base_state,
                         "candidate_destinations": destinations or base_state["candidate_destinations"],
                         "destination_results": []}
                    )
                    accommodation_results = result.get("accommodation_results", [])
                    errors.extend(result.get("errors", []))

                elif tool == "currency_agent":
                    flights_by_dest: dict[str, list] = {}
                    for f in all_flight_results:
                        flights_by_dest.setdefault(f.destination_iata, []).append(f)

                    required_origins = {t.origin_iata for t in query_state.travellers}
                    destination_results = []
                    for iata, dest_flights in flights_by_dest.items():
                        if query_state.search_mode != "specific":
                            covered = {f.origin_iata for f in dest_flights}
                            if not required_origins.issubset(covered):
                                errors.append(f"{iata} dropped — no flights from: {required_origins - covered}")
                                continue
                        dest = DestinationResult(city=iata, iata=iata)
                        dest.flights = dest_flights
                        dest.total_flight_cost_usd = sum(f.price_usd or f.price_local for f in dest_flights)
                        destination_results.append(dest)

                    accom_index = {a.destination_iata: a for a in accommodation_results}
                    for dest in destination_results:
                        accom = accom_index.get(dest.iata)
                        if accom:
                            dest.accommodation = accom
                            dest.total_accommodation_usd = accom.total_usd
                            dest.grand_total_usd = (dest.total_flight_cost_usd or 0.0) + (accom.total_usd or 0.0)

                    result = run_currency_agent({
                        **base_state,
                        "destination_results": destination_results,
                        "flight_results": all_flight_results,
                        "accommodation_results": accommodation_results,
                    })
                    destination_results = result.get("destination_results", destination_results)
                    exchange_rates = result.get("exchange_rates", {})
                    errors.extend(result.get("errors", []))

                elif tool == "ranker_agent":
                    rank_result = run_ranker_agent({"destination_results": destination_results})
                    ranked_destinations = rank_result.get("ranked_destinations", [])

                elif tool == "memory_write":
                    result = run_memory_write({
                        **state,
                        "destination_results": destination_results,
                        "flight_results": all_flight_results,
                        "accommodation_results": accommodation_results,
                        "exchange_rates": exchange_rates,
                        "ranked_destinations": ranked_destinations,
                    })
                    errors.extend(result.get("errors", []))

            except Exception as exc:
                errors.append(f"Step {sid} ({tool}) failed: {exc}")

    return {
        **state,
        "destination_results": destination_results,
        "flight_results": all_flight_results,
        "accommodation_results": accommodation_results,
        "exchange_rates": exchange_rates,
        "ranked_destinations": ranked_destinations,
        "errors": errors,
    }


# ── Conditional edges ─────────────────────────────────────────

def plan_valid(state: dict) -> Literal["execute", "replan"]:
    return "execute" if state.get("rewoo_plan_valid") else "replan"


def replan_or_execute(state: dict) -> Literal["plan", "execute"]:
    if state.get("rewoo_plan_valid") or state.get("replan_count", 0) >= 2:
        return "execute"
    return "plan"