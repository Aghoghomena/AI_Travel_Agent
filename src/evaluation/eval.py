"""
Evaluation Harness — Group Travel Agent

Runs three LangSmith-tracked evaluation experiments:
  1. Ranker accuracy    — T1–T5 scenarios, top-1 accuracy + Kendall's tau
  2. Intent accuracy    — classification rate + field extraction rate
  3. Plan quality       — structural validity + constraint satisfaction + LLM-as-judge

Usage:
    python -m src.evaluation.harness            # all three experiments
    python -m src.evaluation.harness --ranker   # ranker only
    python -m src.evaluation.harness --intent   # intent only
    python -m src.evaluation.harness --plan     # plan only
    python -m src.evaluation.harness --local    # skip LangSmith, print results locally
"""

from __future__ import annotations
import json
import sys
import os
import time
from datetime import datetime
from typing import Any
from langsmith import Client, evaluate as ls_evaluate
from src.utils.config import llm
import random

randomnum = random.randint(1, 100000)
client = Client()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _kendalls_tau(predicted: list[str], ground_truth: list[str]) -> float:
    # Measures how well the ranker's ordering matches the correct order.
    # Counts pairs of destinations: +1 if they're in the right relative order, -1 if swapped.
    # Returns a value from -1 (completely reversed) to +1 (perfect), scaled to 0–1 for display.
    items = [x for x in predicted if x in ground_truth]
    n = len(items)
    if n <= 1:
        return 1.0
    gt_rank = {item: i for i, item in enumerate(ground_truth)}
    pred_rank = {item: i for i, item in enumerate(predicted)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            if (pred_rank[a] < pred_rank[b]) == (gt_rank[a] < gt_rank[b]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def _make_dest(iata: str, flight_usd: float, accom_usd: float):
    # Builds a DestinationResult object from plain cost numbers so the ranker
    # receives the same data structure it gets in production, not raw dicts.
    from src.state import DestinationResult, FlightResult, AccommodationResult
    d = DestinationResult(city=iata, iata=iata)
    d.total_flight_cost_usd = flight_usd
    d.total_accommodation_usd = accom_usd
    d.grand_total_usd = round(flight_usd + accom_usd, 2)
    d.flights = [FlightResult(
        origin_iata="DUB", destination_iata=iata,
        price_local=flight_usd, currency="USD",
        avg_cost_per_night=0, price_usd=flight_usd,
    )]
    d.accommodation = AccommodationResult(
        destination_city=iata, destination_iata=iata,
        price_per_night_local=accom_usd / 7,
        price_per_night_usd=accom_usd / 7,
        currency="USD", nights=7, total_usd=accom_usd,
    )
    return d


# ══════════════════════════════════════════════════════════════════════════════
# 1. RANKER ACCURACY — T1 through T5
# ══════════════════════════════════════════════════════════════════════════════

def create_ranker_dataset() -> list[dict]:
    # The 5 test cases for the ranker, one per traveller count (1–5).
    # Each case has hardcoded flight + accommodation costs and a known correct ranking.
    # Ground truth is just ascending grand_total_usd — cheapest destination first.
    return [
        {   # T1 — 1 traveller, 3 destinations
            "inputs": {
                "scenario_id": "T1",
                "description": "1 traveller, 3 European destinations",
                "destinations": [
                    {"iata": "IST", "flight_usd": 340.0, "accom_usd": 210.0},
                    {"iata": "LIS", "flight_usd": 410.0, "accom_usd": 280.0},
                    {"iata": "AMS", "flight_usd": 520.0, "accom_usd": 340.0},
                ],
            },
            "outputs": {
                "ground_truth_order": ["IST", "LIS", "AMS"],
                "cheapest": "IST",
                "grand_totals": {"IST": 550.0, "LIS": 690.0, "AMS": 860.0},
            },
        },
        {   # T2 — 2 travellers, 4 destinations
            "inputs": {
                "scenario_id": "T2",
                "description": "2 travellers (DUB + LOS), 4 destinations",
                "destinations": [
                    {"iata": "IST", "flight_usd": 820.0,  "accom_usd": 240.0},
                    {"iata": "LIS", "flight_usd": 1050.0, "accom_usd": 330.0},
                    {"iata": "AMS", "flight_usd": 1100.0, "accom_usd": 400.0},
                    {"iata": "BCN", "flight_usd": 960.0,  "accom_usd": 310.0},
                ],
            },
            "outputs": {
                "ground_truth_order": ["IST", "BCN", "LIS", "AMS"],
                "cheapest": "IST",
                "grand_totals": {"IST": 1060.0, "BCN": 1270.0, "LIS": 1380.0, "AMS": 1500.0},
            },
        },
        {   # T3 — 3 travellers, 4 destinations
            "inputs": {
                "scenario_id": "T3",
                "description": "3 travellers (DUB + LOS + LHR), 4 destinations",
                "destinations": [
                    {"iata": "NBO", "flight_usd": 980.0,  "accom_usd": 280.0},
                    {"iata": "ACC", "flight_usd": 860.0,  "accom_usd": 210.0},
                    {"iata": "IST", "flight_usd": 1240.0, "accom_usd": 240.0},
                    {"iata": "LIS", "flight_usd": 1420.0, "accom_usd": 330.0},
                ],
            },
            "outputs": {
                "ground_truth_order": ["ACC", "NBO", "IST", "LIS"],
                "cheapest": "ACC",
                "grand_totals": {"ACC": 1070.0, "NBO": 1260.0, "IST": 1480.0, "LIS": 1750.0},
            },
        },
        {   # T4 — 4 travellers, 3 destinations (close costs)
            "inputs": {
                "scenario_id": "T4",
                "description": "4 travellers, 3 destinations with close costs",
                "destinations": [
                    {"iata": "MAD", "flight_usd": 1680.0, "accom_usd": 420.0},
                    {"iata": "BCN", "flight_usd": 1695.0, "accom_usd": 415.0},
                    {"iata": "LIS", "flight_usd": 1720.0, "accom_usd": 430.0},
                ],
            },
            "outputs": {
                "ground_truth_order": ["MAD", "BCN", "LIS"],
                "cheapest": "MAD",
                "grand_totals": {"MAD": 2100.0, "BCN": 2110.0, "LIS": 2150.0},
            },
        },
        {   # T5 — 5 travellers, 5 destinations
            "inputs": {
                "scenario_id": "T5",
                "description": "5 travellers, 5 destinations — full coverage",
                "destinations": [
                    {"iata": "IST", "flight_usd": 2100.0, "accom_usd": 240.0},
                    {"iata": "NBO", "flight_usd": 1820.0, "accom_usd": 280.0},
                    {"iata": "ACC", "flight_usd": 1650.0, "accom_usd": 210.0},
                    {"iata": "LIS", "flight_usd": 2350.0, "accom_usd": 330.0},
                    {"iata": "AMS", "flight_usd": 2600.0, "accom_usd": 400.0},
                ],
            },
            "outputs": {
                "ground_truth_order": ["ACC", "NBO", "IST", "LIS", "AMS"],
                "cheapest": "ACC",
                "grand_totals": {"ACC": 1860.0, "NBO": 2100.0, "IST": 2340.0, "LIS": 2680.0, "AMS": 3000.0},
            },
        },
    ]


def run_ranker_target(inputs: dict) -> dict:
    # Runs the real ranker agent against one test case.
    # Converts raw cost dicts into DestinationResult objects, calls the ranker,
    # and returns a flat dict (ranked IATA codes + grand totals) the evaluators can read.
    from src.agents.ranker import run_ranker_agent
    dests = [_make_dest(d["iata"], d["flight_usd"], d["accom_usd"])
             for d in inputs["destinations"]]
    result = run_ranker_agent({"destination_results": dests})
    ranked = result.get("ranked_destinations", [])
    return {
        "ranked_iatas":   [d.iata for d in ranked],
        "rank_1":         ranked[0].iata if ranked else None,
        "scenario_id":    inputs["scenario_id"],
        "grand_totals":   {d.iata: d.grand_total_usd for d in ranked},
    }


def eval_ranker_top1_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    # Did the ranker pick the right cheapest destination? Score is 1 (correct) or 0 (wrong).
    correct = outputs.get("rank_1") == reference_outputs.get("cheapest")
    return {"key": "top1_accuracy", "score": int(correct),
            "comment": f"Predicted rank-1: {outputs.get('rank_1')} | Expected: {reference_outputs.get('cheapest')}"}


def eval_ranker_kendall_tau(outputs: dict, reference_outputs: dict) -> dict:
    # Is the full ordering correct? Uses Kendall's tau — a score of 1.0 means every destination is in the right relative order, 0.5 is random noise.
    predicted = outputs.get("ranked_iatas", [])
    ground_truth = reference_outputs.get("ground_truth_order", [])
    tau = _kendalls_tau(predicted, ground_truth)
    return {"key": "kendalls_tau", "score": round((tau + 1) / 2, 4),
            "comment": f"τ={tau:.3f} | predicted={predicted} | truth={ground_truth}"}


def eval_ranker_full_output(outputs: dict, reference_outputs: dict) -> dict:
    # Strictest check: does the entire ranked list match the correct order exactly?
    # Score is 1 only if every destination is in exactly the right position is the position the same gotten from the ranking agent as expected.
    predicted = outputs.get("ranked_iatas", [])
    ground_truth = reference_outputs.get("ground_truth_order", [])
    correct = predicted == ground_truth
    return {"key": "full_ranking_match", "score": int(correct),
            "comment": f"Exact match: {correct}"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. INTENT CLASSIFICATION ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

def intent_dataset() -> list[dict]:
    # 10 test messages covering the three outcomes the intent classifier must handle:
    #   - "ready"      → all info present, can start searching
    #   - "needs_info" → something is missing (origins, duration, etc.)
    #   - "out_of_scope" → not a group travel request at all
    return [
        {
            "inputs": {"message": "Me and my friend want to meet. I'm from Dublin, she's from Lagos. We want to go somewhere in Europe in September for 7 nights, open to anywhere."},
            "outputs": {"expected_status": "ready", "expected_traveller_count": 2, "expected_duration": 7, "expected_mode": "anywhere"},
        },
        {
            "inputs": {"message": "3 of us flying from London, Nairobi, and Amsterdam. We want somewhere warm in Africa for 5 nights in December."},
            "outputs": {"expected_status": "ready", "expected_traveller_count": 3, "expected_duration": 5, "expected_region": "africa"},
        },
        {
            "inputs": {"message": "Can you check flights from Dublin and Lagos to Istanbul in August for 5 nights?"},
            "outputs": {"expected_status": "ready", "expected_traveller_count": 2, "expected_mode": "specific", "expected_destination": "istanbul"},
        },
        {
            "inputs": {"message": "Me and my 2 friends want to meet somewhere in October. Budget $600 per person."},
            "outputs": {"expected_status": "needs_info", "missing_field": "origins"},
        },
        {
            "inputs": {"message": "I'm flying from Dublin and my friend is from Lagos. We want to meet in September."},
            "outputs": {"expected_status": "needs_info", "missing_field": "duration_nights"},
        },
        {
            "inputs": {"message": "Find somewhere cheap to meet."},
            "outputs": {"expected_status": "needs_info", "missing_field": "origins"},
        },
        {
            "inputs": {"message": "Can you book me a hotel in Paris for next weekend?"},
            "outputs": {"expected_status": "out_of_scope"},
        },
        {
            "inputs": {"message": "What is the weather in Dubai in July?"},
            "outputs": {"expected_status": "out_of_scope"},
        },
        {
            "inputs": {"message": "Write me a Python function to sort a list."},
            "outputs": {"expected_status": "out_of_scope"},
        },
        {
            "inputs": {"message": "4 friends from Dublin, Lagos, London, and Accra. We want to meet somewhere in Europe or Africa. Budget $800/person. 6 nights in November. Direct flights only."},
            "outputs": {"expected_status": "ready", "expected_traveller_count": 4, "expected_duration": 6, "expected_budget": 800},
        },
    ]


def run_intent_target(inputs: dict) -> dict:
    # Runs the real intent classifier against one test message.
    # Returns a flat dict of everything the classifier extracted: status, traveller count,
    # duration, search mode, region, budget, and any follow-up question it generated.
    from src.agents.intent_classifier import handle_user_query
    from src.state import QueryState, query_state_to_dict
    result = handle_user_query({
        "user_message": inputs["message"],
        "query_state": query_state_to_dict(QueryState()),
        "conversation_history": [],
        "turn_count": 0,
    })
    qs = result.get("query_state", {})
    travellers = qs.get("travellers", []) if isinstance(qs, dict) else []
    return {
        "intent_status":    str(result.get("intent_status", "")).lower().replace("intentstatus.", ""),
        "traveller_count":  len(travellers),
        "duration_nights":  qs.get("duration_nights") if isinstance(qs, dict) else None,
        "search_mode":      qs.get("search_mode") if isinstance(qs, dict) else None,
        "region":           qs.get("region_preferences") if isinstance(qs, dict) else None,
        "max_budget_usd":   qs.get("max_budget_usd") if isinstance(qs, dict) else None,
        "destination_locs": qs.get("destination_locations", []) if isinstance(qs, dict) else [],
        "elicitation_q":    result.get("elicitation_question", ""),
        "elicitation_complete": result.get("elicitation_complete", False),
    }


def intent_status_correctness(outputs: dict, reference_outputs: dict) -> dict:
    # Did the classifier return the right status (ready / needs_info / out_of_scope)?
    # Score is 1 or 0. This is checking do we have enough information to run the agent so we dont end up with errors
    predicted = outputs.get("intent_status", "").replace("intentstatus.", "")
    expected  = reference_outputs.get("expected_status", "")
    correct   = expected in predicted or predicted == expected
    return {"key": "status_correct", "score": int(correct),
            "comment": f"predicted={predicted!r} expected={expected!r}"}


def eval_traveller_count(outputs: dict, reference_outputs: dict) -> dict:
    # Did the classifier detect at least as many travellers as expected?
    # Uses >= so that detecting extra travellers doesn't fail the test.
    expected = reference_outputs.get("expected_traveller_count")
    if expected is None:
        return {"key": "traveller_count_correct", "score": 1.0, "comment": "N/A"}
    correct = outputs.get("traveller_count", 0) >= expected
    return {"key": "traveller_count_correct", "score": int(correct),
            "comment": f"predicted={outputs.get('traveller_count')} expected≥{expected}"}


def eval_user_input_field_extraction(outputs: dict, reference_outputs: dict) -> dict:
    # What fraction of the expected fields (duration, mode, region, budget) did the
    # classifier correctly pull out of the message? Returns a score from 0.0 to 1.0.
    checks, hits = 0, 0
    if (d := reference_outputs.get("expected_duration")):
        checks += 1
        if outputs.get("duration_nights") == d:
            hits += 1
    if (m := reference_outputs.get("expected_mode")):
        checks += 1
        if (outputs.get("search_mode") or "").lower() == m.lower():
            hits += 1
    if (r := reference_outputs.get("expected_region")):
        checks += 1
        if r.lower() in (outputs.get("region") or "").lower():
            hits += 1
    if (b := reference_outputs.get("expected_budget")):
        checks += 1
        budget = outputs.get("max_budget_usd")
        if budget and abs(float(budget) - float(b)) < 50:
            hits += 1
    score = hits / checks if checks > 0 else 1.0
    return {"key": "field_extraction_rate", "score": round(score, 3),
            "comment": f"{hits}/{checks} fields correctly extracted"}


def eval_elicitation_llm_judge(outputs: dict, reference_outputs: dict) -> dict:
    # For "needs_info" cases: asks an LLM to judge whether the follow-up question
    # the classifier generated actually asks for the right missing piece of info.
    # Score is 1 (relevant question) or 0 (wrong question / no question).
    missing = reference_outputs.get("missing_field")
    if not missing or reference_outputs.get("expected_status") != "needs_info":
        return {"key": "elicitation_relevance", "score": 1.0, "comment": "N/A (not a NEEDS_INFO case)"}

    question = outputs.get("elicitation_q", "")
    if not question:
        return {"key": "elicitation_relevance", "score": 0.0, "comment": "No question generated"}

    from src.utils.config import llm
    prompt = f"""You are evaluating an AI travel agent's elicitation question.
            The agent knows a user wants group travel but is missing: {missing}
            The agent asked: "{question}"
            Score 1 if the question specifically asks for {missing} (or closely related info).
            Score 0 if the question asks for something else or is off-topic.
            Respond with ONLY a JSON object: {{"score": 0 or 1, "reason": "brief explanation"}}"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        score = int(bool(parsed.get("score", 0)))
        return {"key": "elicitation_relevance", "score": score,
                "comment": parsed.get("reason", "")}
    except Exception as e:
        return {"key": "elicitation_relevance", "score": 0.5,
                "comment": f"Judge error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLAN GENERATION QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def _plan_dataset() -> list[dict]:
    # 5 test scenarios for plan generation, covering different combinations of
    # traveller count, search mode, budget constraints, and direct-flight requirements.
    # Ground truth is which tool steps should appear in the generated XML plan.
    def _t(name, city, iata):
        return {"name": name, "origin_city": city, "origin_iata": iata}

    return [
        {
            "inputs": {
                "scenario_id": "P1",
                "description": "2 travellers, anywhere, no constraints",
                "travellers": [_t("Alice", "Dublin", "DUB"), _t("Ben", "Lagos", "LOS")],
                "travel_month": "September", "duration_nights": 7,
                "search_mode": "anywhere", "region_preferences": None,
                "max_budget_usd": None, "direct_flights_only": False,
                "accommodation_needed": True,
            },
            "outputs": {"expected_traveller_steps": 2, "has_budget_filter": False, "has_direct_filter": False},
        },
        {
            "inputs": {
                "scenario_id": "P2",
                "description": "3 travellers, Europe, with budget",
                "travellers": [_t("Alice","Dublin","DUB"), _t("Ben","Lagos","LOS"), _t("Cara","London","LHR")],
                "travel_month": "October", "duration_nights": 5,
                "search_mode": "region", "region_preferences": "europe",
                "max_budget_usd": 1000, "direct_flights_only": False,
                "accommodation_needed": True,
            },
            "outputs": {"expected_traveller_steps": 3, "has_budget_filter": True, "has_direct_filter": False},
        },
        {
            "inputs": {
                "scenario_id": "P3",
                "description": "2 travellers, direct flights only",
                "travellers": [_t("Alice","Dublin","DUB"), _t("Ben","Nairobi","NBO")],
                "travel_month": "December", "duration_nights": 10,
                "search_mode": "anywhere", "region_preferences": None,
                "max_budget_usd": None, "direct_flights_only": True,
                "accommodation_needed": True,
            },
            "outputs": {"expected_traveller_steps": 2, "has_budget_filter": False, "has_direct_filter": True},
        },
        {
            "inputs": {
                "scenario_id": "P4",
                "description": "4 travellers, Africa, budget + direct",
                "travellers": [_t("A","Dublin","DUB"),_t("B","Lagos","LOS"),_t("C","London","LHR"),_t("D","Accra","ACC")],
                "travel_month": "July", "duration_nights": 6,
                "search_mode": "region", "region_preferences": "africa",
                "max_budget_usd": 800, "direct_flights_only": True,
                "accommodation_needed": True,
            },
            "outputs": {"expected_traveller_steps": 4, "has_budget_filter": True, "has_direct_filter": True},
        },
        {
            "inputs": {
                "scenario_id": "P5",
                "description": "2 travellers, specific destination",
                "travellers": [_t("Alice","Dublin","DUB"), _t("Ben","Lagos","LOS")],
                "travel_month": "August", "duration_nights": 5,
                "search_mode": "specific", "region_preferences": None,
                "max_budget_usd": None, "direct_flights_only": False,
                "accommodation_needed": True,
            },
            "outputs": {"expected_traveller_steps": 2, "has_budget_filter": False, "has_direct_filter": False},
        },
    ]


def run_plan_target(inputs: dict) -> dict:
    # Runs the real orchestrator plan_node against one test scenario.
    # Parses the XML plan it generates to count which tools appear and how many times.
    # Returns validity, tool counts, and any errors so the evaluators can score it.
    import xml.etree.ElementTree as ET
    from src.agents.orchestrator import plan_node, validate_plan
    from src.state import Traveller, query_state_to_dict, QueryState
    from src.memory.semantic import SemanticMemory

    travellers = [Traveller(**t) for t in inputs["travellers"]]
    qs = QueryState()
    qs.travellers        = travellers
    qs.travel_month      = inputs["travel_month"]
    qs.duration_nights   = inputs["duration_nights"]
    qs.search_mode       = inputs["search_mode"]
    qs.region_preferences = inputs.get("region_preferences")
    qs.max_budget_usd    = inputs.get("max_budget_usd")
    qs.direct_flights_only = inputs.get("direct_flights_only", False)
    qs.accommodation_needed = inputs.get("accommodation_needed", True)

    state = {
        "query_state": query_state_to_dict(qs),
        "prioritised_destinations": [],
        "errors": [],
        "hitl_plan_feedback": None,
        "candidate_destinations": [],
        "rewoo_plan": None,
        "rewoo_plan_valid": False,
        "replan_count": 0,
        "hitl_plan_approved": None,
        "hitl_replan_complete": False,
    }

    try:
        result = plan_node(state)
        plan_xml = result.get("rewoo_plan", "")
        is_valid, error_msg = validate_plan(plan_xml, travellers, result.get("candidate_destinations", []))

        tool_counts: dict[str, int] = {}
        if plan_xml:
            try:
                root = ET.fromstring(plan_xml.strip())
                for step in root.findall("step"):
                    t = step.get("tool", "")
                    tool_counts[t] = tool_counts.get(t, 0) + 1
            except ET.ParseError:
                pass

        return {
            "scenario_id":       inputs["scenario_id"],
            "plan_xml":          plan_xml,
            "is_valid":          is_valid,
            "validation_error":  error_msg,
            "tool_counts":       tool_counts,
            "traveller_steps":   tool_counts.get("traveller_agent", 0),
            "has_budget_filter": "budget_filter" in tool_counts,
            "has_direct_filter": "direct_flights_filter" in tool_counts,
            "errors":            result.get("errors", []),
        }
    except Exception as exc:
        return {
            "scenario_id": inputs["scenario_id"],
            "plan_xml": "", "is_valid": False, "tool_counts": {},
            "traveller_steps": 0, "has_budget_filter": False,
            "has_direct_filter": False, "errors": [str(exc)],
            "validation_error": str(exc),
        }


def eval_plan_structural_validity(outputs: dict, reference_outputs: dict) -> dict:
    # Is the generated XML plan structurally valid? Score is 1 (passes validation) or 0.
    valid = outputs.get("is_valid", False)
    return {"key": "structural_validity", "score": int(valid),
            "comment": outputs.get("validation_error", "OK") if not valid else "Valid XML plan"}


def eval_plan_traveller_count(outputs: dict, reference_outputs: dict) -> dict:
    # Did the plan include exactly one traveller_agent step per traveller?
    # If there are 3 travellers, the plan must call traveller_agent exactly 3 times this is very important
    expected = reference_outputs.get("expected_traveller_steps", 0)
    actual   = outputs.get("traveller_steps", 0)
    correct  = actual == expected
    return {"key": "traveller_step_count", "score": int(correct),
            "comment": f"expected={expected} actual={actual}"}


def eval_plan_constraint_satisfaction(outputs: dict, reference_outputs: dict) -> dict:
    # Did the plan include the right filter steps for the user's constraints?
    # Checks: budget_filter included iff budget was set, direct_flights_filter included iff direct-only was requested.
    checks, hits = 0, 0
    if reference_outputs.get("has_budget_filter") is not None:
        checks += 1
        if outputs.get("has_budget_filter") == reference_outputs["has_budget_filter"]:
            hits += 1
    if reference_outputs.get("has_direct_filter") is not None:
        checks += 1
        if outputs.get("has_direct_filter") == reference_outputs["has_direct_filter"]:
            hits += 1
    score = hits / checks if checks > 0 else 1.0
    return {"key": "constraint_satisfaction", "score": round(score, 3),
            "comment": f"{hits}/{checks} constraints correctly reflected in plan"}


def eval_plan_llm_judge(outputs: dict, reference_outputs: dict) -> dict:
    # Asks an LLM to score the overall quality of the plan on a 0.0–1.0 scale.
    # Judges three things: are dependencies in the right order (memory_read first, ranker after
    # currency, memory_write last), is tool selection correct, and is the dependency graph coherent?
    plan_xml = outputs.get("plan_xml", "")
    if not plan_xml or not outputs.get("is_valid"):
        return {"key": "plan_quality_llm", "score": 0.0, "comment": "Invalid or empty plan"}

    from src.utils.config import llm
    prompt = f"""You are evaluating the quality of an AI agent's execution plan for a group travel search.

                The plan is:
                {plan_xml}

                Context:
                - Traveller steps expected: {reference_outputs.get('expected_traveller_steps')}
                - Budget filter needed: {reference_outputs.get('has_budget_filter')}
                - Direct flights filter needed: {reference_outputs.get('has_direct_filter')}

                Evaluate on a scale 0.0–1.0 considering:
                1. Are all dependencies logically ordered (memory_read first, ranker after currency, memory_write last)?
                2. Does the tool selection match the constraints?
                3. Is the dependency graph acyclic and coherent?

                Respond ONLY with JSON: {{"score": float, "reason": "brief explanation"}}"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
        return {"key": "plan_quality_llm", "score": round(score, 3),
                "comment": parsed.get("reason", "")}
    except Exception as e:
        return {"key": "plan_quality_llm", "score": 0.5, "comment": f"Judge error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ══════════════════════════════════════════════════════════════════════════════

def _run_local(name: str, dataset: list[dict], target, evaluators: list) -> dict:
    # Runs an experiment locally without LangSmith. Loops over each test case,
    # calls the target function, runs all evaluators, and collects scores into a dict.
    results = []
    for ex in dataset:
        outputs = target(ex["inputs"])
        ref     = ex.get("outputs", {})
        scores  = {}
        for ev in evaluators:
            try:
                r = ev(outputs, ref)
                scores[r["key"]] = {"score": r["score"], "comment": r.get("comment", "")}
            except Exception as e:
                scores[ev.__name__] = {"score": 0.0, "comment": str(e)}
        results.append({"inputs": ex["inputs"], "outputs": outputs, "scores": scores})
    return {"experiment": name, "results": results}


def run_langsmith(name: str, dataset: list[dict], target, evaluators: list,
                   experiment_prefix: str) -> Any:
    # Runs an experiment via LangSmith so results are logged to the cloud dashboard.
    # Identical logic to _run_local but LangSmith handles the loop, concurrency, and storage.

    dataset_name = f"{experiment_prefix}-{randomnum}"
    # Create the dataset in LangSmith if it doesn't already exist.
    ls_dataset = client.create_dataset(dataset_name=dataset_name, description=name)
    client.create_examples(
        inputs=[ex["inputs"] for ex in dataset],
        outputs=[ex.get("outputs", {}) for ex in dataset],
        dataset_id=ls_dataset.id,
    )
    return ls_evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={"experiment": name},
        max_concurrency=1,
    )


def print_local_results(result: dict) -> None:
    # Pretty-prints the results of a local run: per-scenario scores in colour,
    # then a summary table with average / min / max for each metric.
    _G = "\033[32m"; _R = "\033[31m"; _B = "\033[1m"; _X = "\033[0m"
    print(f"\n{_B}{'─'*68}{_X}")
    print(f"{_B}{result['experiment']}{_X}")
    print(f"{'─'*68}")
    all_metric_keys = sorted({k for r in result["results"] for k in r["scores"]})
    totals = {k: [] for k in all_metric_keys}
    for r in result["results"]:
        sid = r["inputs"].get("scenario_id", r["inputs"].get("message", "")[:30])
        print(f"\n  Scenario: {sid}")
        for key in all_metric_keys:
            s = r["scores"].get(key, {})
            score = s.get("score", "–")
            color = _G if isinstance(score, (int,float)) and score >= 0.8 else (_R if isinstance(score,(int,float)) and score < 0.5 else "")
            print(f"    {key:<35} {color}{score}{_X}  {s.get('comment','')[:60]}")
            if isinstance(score, (int, float)):
                totals[key].append(score)
    print(f"\n{'─'*68}")
    print(f"  {'Metric':<35} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*6}")
    for key in all_metric_keys:
        vals = totals[key]
        if vals:
            avg = sum(vals)/len(vals)
            color = _G if avg >= 0.8 else (_R if avg < 0.5 else "")
            print(f"  {key:<35} {color}{avg:>6.3f}{_X}  {min(vals):>6.3f}  {max(vals):>6.3f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ranker(use_langsmith: bool = True) -> Any:
    # Entry point for the ranker experiment. Wires up the dataset, target function,
    # and three evaluators, then runs via LangSmith or locally depending on the flag.
    print(f"Ranker Evaluator started on langsmith {use_langsmith}")
    dataset    = create_ranker_dataset()
    evaluators = [eval_ranker_top1_accuracy, eval_ranker_kendall_tau, eval_ranker_full_output]
    if use_langsmith:
        return run_langsmith("Ranker Accuracy T1–T5", dataset, run_ranker_target,
                              evaluators, "eval-ranker")
    result = _run_local("Ranker Accuracy T1–T5", dataset, run_ranker_target, evaluators)
    print_local_results(result)
    return result


def evaluate_intent(use_langsmith: bool = True) -> Any:
    # Entry point for the intent classification experiment. Same wiring pattern as
    # evaluate_ranker but uses the intent dataset and its four evaluators.
    dataset    = intent_dataset()
    evaluators = [intent_status_correctness, eval_traveller_count,
                  eval_user_input_field_extraction, eval_elicitation_llm_judge]
    if use_langsmith:
        return run_langsmith("Intent Classification", dataset, run_intent_target,
                              evaluators, "eval-intent")
    result = _run_local("Intent Classification", dataset, run_intent_target, evaluators)
    print_local_results(result)
    return result


def evaluate_plan(use_langsmith: bool = True) -> Any:
    # Entry point for the plan generation experiment. Same wiring pattern but uses
    # the plan dataset and its four evaluators (structural validity, traveller count,
    # constraint satisfaction, LLM judge).
    dataset    = _plan_dataset()
    evaluators = [eval_plan_structural_validity, eval_plan_traveller_count,
                  eval_plan_constraint_satisfaction, eval_plan_llm_judge]
    if use_langsmith:
        return run_langsmith("Plan Generation Quality", dataset, run_plan_target,
                              evaluators, "eval_plan_agent")
    result = _run_local("Plan Generation Quality", dataset, run_plan_target, evaluators)
    print_local_results(result)
    return result


def run_all(use_langsmith: bool = True) -> None:
    # Runs all three experiments in sequence (ranker → intent → plan), timing each one.
    # If using LangSmith, prints a link to the dashboard at the end.
    experiments = []
    print("\nRunning evaluation harness…\n")
    for name, fn in [
        ("Ranker Accuracy (T1–T5)", lambda: evaluate_ranker(use_langsmith)),
        ("Intent Classification",   lambda: evaluate_intent(use_langsmith)),
        ("Plan Generation Quality", lambda: evaluate_plan(use_langsmith)),
    ]:
        print(f"  ▶ {name}")
        start = time.perf_counter()
        r = fn()
        elapsed = time.perf_counter() - start
        experiments.append((name, r, elapsed))
        print(f"    Done in {elapsed:.1f}s")

    if use_langsmith:
        print("\n✓ Results logged to LangSmith. View at https://smith.langchain.com\n")


if __name__ == "__main__":
    print(f"Starting Evaluator")
    use_ls = True
    # evaluate_ranker(use_ls)
    # evaluate_intent(use_ls) evaluate_plan(use_ls)
    run_all(use_ls)
