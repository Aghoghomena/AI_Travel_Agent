"""
Memory Agent.
 
Wraps EpisodicMemory, ProceduralMemory and SemanticMemory into a
single agent interface with two modes:
 
READ  (pre-search)  — check episodic cache for group search,
                      check procedural for destination prioritisation,
                      surface past searches conversationally.
 
WRITE (post-search) — store completed search to episodic,
                      record winner to procedural,
                      log HITL confirmations to audit log.
 
LangGraph nodes:
  run_memory_read(state)  -> dict
  run_memory_write(state) -> dict
"""


import json
from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, START, END
 
from src.memory.db import get_db_connection, initialize_db
from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.semantic import SemanticMemory
from src.state import QueryState, MemoryReadState, MemoryWriteState, query_state_from_dict, query_state_to_dict, DestinationResult


# ── Singletons ────────────────────────────────────────────────
initialize_db()
episodic   = EpisodicMemory()
procedural = ProceduralMemory()
semantic   = SemanticMemory()

 
 
# ── Helpers ───────────────────────────────────────────────────
 
def _origins(query_state: QueryState) -> list[str]:
    return [t.origin_iata for t in query_state.travellers if t.origin_iata]
 
 
def _outbound_date(query_state: QueryState) -> str:
    return query_state.outbound_date or query_state.travel_month or ""

# ── READ nodes ────────────────────────────────────────────────
 
def check_episodic_node(state: MemoryReadState) -> MemoryReadState:
    """
    Node 1 (READ): Checks episodic cache for a prior full group search
    matching same origins + destinations + date + nights.
    """

    # print (f"state at check_episodic_node {state}")
    query_state = state.get("query_state", QueryState())
    origins = _origins(query_state)
    outbound_date = _outbound_date(query_state)
    nights = query_state.duration_nights or 1
    candidates = query_state.candidate_destinations
 
    if not origins or not outbound_date:
        return {**state, "cache_hit": False, "cached_result": None}
 
    dest_iatas = [
        d.iata if hasattr(d, "iata") else str(d)
        for d in candidates
    ]
 
    cached = episodic.get_full_group_search(origins, dest_iatas, outbound_date, nights,  region_preferences=query_state.region_preferences or "")
    # print(f"output of search episodic memory {cached}")
    if cached:
        return {**state, "cache_hit": True, "cached_result": cached}
 
    return {**state, "cache_hit": False, "cached_result": None}

def check_procedural_node(state: MemoryReadState) -> MemoryReadState:
    """
    Node 2 (READ): Gets historically winning destinations for this
    origin combination — used to prioritise candidate evaluation order.
    """
    query_state = state.get("query_state", QueryState())
    origins = _origins(query_state)
 
    if not origins:
        return {**state, "prioritised_destinations": []}
 
    top = procedural.get_top_destinations_for_origin_pattern(origins, top_n=5)
    prioritised = [d.get("destination", d.get("iata", "")) for d in top]
 
    return {**state, "prioritised_destinations": prioritised}

def surface_past_searches_node(state: MemoryReadState) -> MemoryReadState:
    """
    Node 3 (READ): Surfaces past searches for this group.
    Only runs on cache miss to give context.
    """
    if state.get("cache_hit"):
        return {**state, "past_searches_summary": None}
 
    query_state = state.get("query_state", QueryState())
    origins = _origins(query_state)
 
    if not origins:
        return {**state, "past_searches_summary": None}
 
    past = episodic.get_past_searches(origins)
    if not past:
        return {**state, "past_searches_summary": None}
 
    lines = [
        f"  {p['destinations'].upper()} on {p['outbound_date']} — ${p['total_usd']:.0f} total"
        for p in past
    ]
    summary = f"Found {len(past)} past search(es) for this group:\n" + "\n".join(lines)
    return {**state, "past_searches_summary": summary}


# ── WRITE nodes ───────────────────────────────────────────────
 
def write_episodic_node(state: MemoryWriteState) -> MemoryWriteState:
    """
    Node 1 (WRITE): Stores individual flight legs and the full group
    search result to episodic memory.
    """
    
    # print(f" \nthe query state at write episodic {QueryState} \n")
    # print(f"\n the state at write episodic {state}\n")
    query_state  = state.get("query_state", QueryState())
    flight_results       = state.get("flight_results", [])
   # origins      = _origins(query_state)
    origins = list({f.origin_iata.lower() for f in flight_results if f.origin_iata})
    outbound_date = _outbound_date(query_state)
    accommodation_results = state.get("accommodation_results", [])
    nights = accommodation_results[0].nights if accommodation_results else (query_state.duration_nights or 0)
    exchange_rates = state.get("exchange_rates", {})
    ranked       = state.get("ranked_destinations", [])
    activities_by_dest = state.get("activities", {})
    errors       = list(state.get("errors", []))
    region = query_state.region_preferences or ""

    # print (f"the origins {origins} at write_episodic_node \n")

 
    # Cache individual flight legs
    for f in flight_results:
        try:
            episodic.set_flight_leg(f)
        except Exception as e:
            errors.append(f"[memory] flight leg cache write failed: {e}")
 
    # Cache full group search per destination
    accom_index = {a.destination_iata: a for a in accommodation_results}
 
    for dest in ranked:
        dest_flights = [f for f in flight_results if f.destination_iata == dest.iata]
        accom = accom_index.get(dest.iata)
        if not dest_flights or not accom:
            continue
        try:
            episodic.set_group_search(
                origins=origins,
                destination=dest.iata,
                outbound_date=outbound_date,
                return_date=query_state.return_date or "",
                duration_nights=nights,
                flights=dest_flights,
                accommodation=accom,
                total_usd=dest.grand_total_usd or 0.0,
                exchange_rates=exchange_rates,
                activities=activities_by_dest.get(dest.iata),
                region_preferences=region
            )
        except Exception as e:
            errors.append(f"[memory] group search cache write failed for {dest.iata}: {e}")
 
    return {**state, "errors": errors}

def write_procedural_node(state: MemoryWriteState) -> MemoryWriteState:
    """
    Node 2 (WRITE): Records the winning destination to procedural memory.
    """
    query_state = state.get("query_state", QueryState())
    origins     = _origins(query_state)
    ranked      = state.get("ranked_destinations", [])
    errors      = list(state.get("errors", []))
 
    if not ranked or not origins:
        return state
 
    winner    = ranked[0]
    all_iatas = [d.iata for d in ranked]
    saving    = 0.0
    if len(ranked) >= 2:
        saving = (ranked[1].grand_total_usd or 0.0) - (winner.grand_total_usd or 0.0)
 
    try:
        procedural.record_winner(
            origins=origins,
            winning_destination=winner.iata,
            all_destinations=all_iatas,
            savings_vs_second_usd=saving,
        )
    except Exception as e:
        errors.append(f"[memory] procedural write failed: {e}")
 
    return {**state, "errors": errors}
 

def log_hitl_node(state: MemoryWriteState) -> MemoryWriteState:
    """
    Node 3 (WRITE): Logs HITL checkpoint outcomes to the audit log.
    """
    session_id = state.get("session_id", "unknown")
    errors     = list(state.get("errors", []))
 
    checkpoints = [
        state.get("hitl_checkpoint_1"),
        state.get("hitl_checkpoint_2"),
    ]
 
    conn = get_db_connection()
    for cp in checkpoints:
        if cp is None:
            continue
        try:
            correction_json = None
            if cp.correction:
                correction_json = json.dumps(cp.correction)
 
            conn.execute(
                """INSERT INTO hitl_audit_log
                   (session_id, checkpoint_id, message,
                    user_response, correction_json, logged_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id,
                    cp.checkpoint_id,
                    cp.message,
                    cp.user_response,
                    correction_json,
                    datetime.utcnow().isoformat(),
                ),
            )
        except Exception as e:
            errors.append(f"[memory] HITL log failed: {e}")
 
    conn.commit()
    conn.close()
    return {**state, "errors": errors}

# ── Build READ graph ──────────────────────────────────────────
 
def build_memory_read_graph():
    graph = StateGraph(MemoryReadState)
 
    graph.add_node("check_episodic",  check_episodic_node)
    graph.add_node("check_procedural", check_procedural_node)
    graph.add_node("surface_past",    surface_past_searches_node)
 
    graph.add_edge(START, "check_episodic")
    graph.add_edge("check_episodic",   "check_procedural")
    graph.add_edge("check_procedural", "surface_past")
    graph.add_edge("surface_past",     END)
 
    return graph.compile()
 
 
# ── Build WRITE graph ─────────────────────────────────────────
 
def build_memory_write_graph():
    graph = StateGraph(MemoryWriteState)
 
    graph.add_node("write_episodic",   write_episodic_node)
    graph.add_node("write_procedural", write_procedural_node)
    graph.add_node("log_hitl",         log_hitl_node)
 
    graph.add_edge(START,             "write_episodic")
    graph.add_edge("write_episodic",  "write_procedural")
    graph.add_edge("write_procedural", "log_hitl")
    graph.add_edge("log_hitl",        END)
 
    return graph.compile()
 
 
memory_read_agent  = build_memory_read_graph()
memory_write_agent = build_memory_write_graph()
 
try:
    memory_read_agent.get_graph().draw_mermaid_png(output_file_path="memory_read_agent.png")
    memory_write_agent.get_graph().draw_mermaid_png(output_file_path="memory_write_agent.png")
    print("\nGraphs saved as memory_read_agent.png and memory_write_agent.png")
except Exception as e:
    print(f"\nCould not save PNG: {e}")

# ── Entry points ──────────────────────────────────────────────
 
def run_memory_read(state: dict) -> dict:
    """
    Pre-search: check caches, prioritise destinations, surface past searches.
    Also injects memory instances into state for downstream agents.
    """
    # print(f"\n Running memory read with initial state: {state}")
    result = memory_read_agent.invoke({
        "query_state":              query_state_from_dict(state.get("query_state", {})),
        "session_id":               state.get("session_id", ""),
        "cache_hit":                False,
        "cached_result":            None,
        "prioritised_destinations": [],
        "past_searches_summary":    None,
        "errors":                   [],
        

    })

    ranked_from_cache = []
    if result.get("cache_hit"):
        cached_list = result.get("cached_result", [])
        for i, entry in enumerate(cached_list):
            iata = entry["destinations"][0].upper()
            ranked_from_cache.append(DestinationResult(
                city=iata,
                iata=iata,
                grand_total_usd=entry.get("total_usd"),
                rank=i + 1,
            ))

 
    return {
        "episodic_cache_hit":         result["cache_hit"],
        "cached_result":              result.get("cached_result"),
        "ranked_destinations":        ranked_from_cache,
        "prioritised_destinations":   result.get("prioritised_destinations", []),
        "past_searches_summary":      result.get("past_searches_summary"),
        "errors":                     result.get("errors", []),
        "query_state":                query_state_to_dict(result.get("query_state", QueryState())),
    }
 
 
def run_memory_write(state: dict) -> dict:
    """Post-search: persist results, record winner, log HITL checkpoints."""
    qs_raw = state.get("query_state", {})
    outbound_date = (
        (qs_raw.get("outbound_date") or qs_raw.get("travel_month") or "")
        if isinstance(qs_raw, dict)
        else (getattr(qs_raw, "outbound_date", None) or getattr(qs_raw, "travel_month", None) or "")
    )
    result = memory_write_agent.invoke({
        "query_state":           query_state_from_dict(state.get("query_state", {})),
        "session_id":            state.get("session_id", ""),
        "ranked_destinations":   state.get("ranked_destinations", []),
        "flight_results":        state.get("flight_results", []),
        "accommodation_results": state.get("accommodation_results", []),
        "exchange_rates":        state.get("exchange_rates", {}),
        "hitl_checkpoint_1":     state.get("hitl_checkpoint_1"),
        "hitl_checkpoint_2":     state.get("hitl_checkpoint_2"),
        "outbound_date":         outbound_date,
        "errors":                [],
    })
 
    return {
        **state,
        "memory_write_complete": True,
        "errors":                result.get("errors", []),
    }