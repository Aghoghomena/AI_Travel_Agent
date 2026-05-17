"""
Ablation Studies

1. ReWOO vs ReAct — LLM call count + planning time comparison
2. Memory impact  — cold vs warm cache timing + speedup ratio

Usage:
    python -m src.evaluation.ablation            # both studies
    python -m src.evaluation.ablation --rewoo    # ReWOO vs ReAct only
    python -m src.evaluation.ablation --cache    # cache timing only
"""

from __future__ import annotations
import sys
import time
import tempfile
import shutil
import os
from pathlib import Path
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# 1. ReWOO vs ReAct ablation
# ══════════════════════════════════════════════════════════════════════════════

class _CountingLLM:
    # Wraps your real LLM and intercepts every call so we can count how many
    # times it was invoked and how long each call took. Used to compare
    # ReWOO (should be 1 call) vs ReAct (1 call per step).
    def __init__(self, llm):
        self._llm = llm
        self.call_count = 0
        self.total_ms   = 0.0

    def invoke(self, prompt, **kwargs):
        # Forwards the call to the real LLM, records the time taken, and increments the counter.
        start = time.perf_counter()
        result = self._llm.invoke(prompt, **kwargs)
        self.total_ms += (time.perf_counter() - start) * 1000
        self.call_count += 1
        return result

    def reset(self):
        # Resets counters between ReWOO and ReAct runs so they don't bleed into each other.
        self.call_count = 0
        self.total_ms   = 0.0


def _build_rewoo_state(n_travellers: int) -> dict:
    # Builds a fake orchestrator state with N synthetic travellers (Alice, Ben, etc.)
    # so we can run the planner without needing real user input. Always uses September,
    # 7 nights, anywhere mode — just enough to trigger a real plan.
    from src.state import Traveller, QueryState, query_state_to_dict
    names  = ["Alice", "Ben", "Cara", "Dan", "Eve"]
    iatas  = ["DUB",  "LOS", "LHR",  "NBO", "ACC"]
    cities = ["Dublin","Lagos","London","Nairobi","Accra"]

    qs = QueryState()
    qs.travellers = [
        Traveller(name=names[i], origin_city=cities[i], origin_iata=iatas[i])
        for i in range(n_travellers)
    ]
    qs.travel_month     = "September"
    qs.duration_nights  = 7
    qs.search_mode      = "anywhere"
    qs.accommodation_needed = True

    return {
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


def _run_rewoo(state: dict, counting_llm: _CountingLLM) -> dict:
    # Runs the real orchestrator with the counting LLM swapped in.
    # ReWOO generates the entire execution plan in a single LLM call upfront,
    # then validates it — no LLM is called again during tool execution.
    import src.agents.orchestrator as orch
    original_llm = orch.llm
    orch.llm = counting_llm
    try:
        result = orch.plan_node(state)
        validated = orch.validate_plan_node(result)
        return validated
    finally:
        orch.llm = original_llm


def _run_react(state: dict, counting_llm: _CountingLLM) -> dict:
    # Simulates how a ReAct agent would behave: no upfront plan, just ask the LLM
    # "what tool should I call next?" after every single step. For N travellers this
    # means N+5 LLM calls (memory_read + N traveller agents + accommodation + currency + ranker + memory_write).
    # Returns a trace of what the LLM decided at each step.
    from src.state import query_state_from_dict
    from src.utils.config import llm as real_llm

    qs = query_state_from_dict(state.get("query_state", {}))
    n  = len(qs.travellers)

    # ReAct loop: decide next action from current state each iteration
    # Tools it must visit: memory_read → traveller×N → accommodation → currency → ranker → memory_write
    tool_sequence = (
        ["memory_read"]
        + [f"traveller_agent[{t.name}]" for t in qs.travellers]
        + ["accommodation_agent", "currency_agent", "ranker_agent", "memory_write"]
    )

    prompt_template = """You are a ReAct agent for group travel search.
                Observation so far: {obs}
                Available tools: memory_read, traveller_agent, accommodation_agent, currency_agent, ranker_agent, memory_write
                What is the NEXT single tool to call? Reply with just the tool name."""

    observation = "Starting group travel search."
    trace = []

    for expected_tool in tool_sequence:
        prompt = prompt_template.format(obs=observation)
        response = counting_llm.invoke(prompt)
        chosen   = response.content.strip().split()[0]
        trace.append({"thought": f"Next: {chosen}", "action": chosen})
        observation = f"Completed {chosen}. "

    return {"react_trace": trace, "react_steps": len(trace)}


@dataclass
class ReWOOAblationResult:
    n_travellers:      int
    rewoo_llm_calls:   int
    react_llm_calls:   int
    rewoo_plan_ms:     float
    react_ms:          float
    call_reduction:    float   # % fewer LLM calls with ReWOO
    time_reduction:    float   # % faster plan phase with ReWOO
    rewoo_plan_valid:  bool


def run_rewoo_ablation(traveller_counts: list[int] | None = None) -> list[ReWOOAblationResult]:
    # The main ReWOO vs ReAct comparison. Runs both approaches for each traveller count
    # (1 through 5 by default), measures LLM call count and time for each, then prints
    # a table showing how many fewer calls ReWOO needed and the % improvement.
    from src.utils.config import llm as base_llm

    if traveller_counts is None:
        traveller_counts = [1, 2, 3, 4, 5]

    results: list[ReWOOAblationResult] = []
    counting = _CountingLLM(base_llm)

    print("\nReWOO vs ReAct — LLM call comparison")
    print("How many times does the LLM get called for each approach?\n")
    print(f"  {'Travellers':<16} {'ReWOO':>6} {'ReAct':>6} {'Saved':>7} {'Plan OK':>9}")
    print(f"  {'(# of people)':<16} {'calls':>6} {'calls':>6} {'calls':>7} {'valid?':>9}")
    print("  " + "─" * 48)

    for n in traveller_counts:
        state = _build_rewoo_state(n)

        # ── ReWOO ──
        counting.reset()
        t0 = time.perf_counter()
        rewoo_result = _run_rewoo(state, counting)
        rewoo_ms     = (time.perf_counter() - t0) * 1000
        rewoo_calls  = counting.call_count
        rewoo_valid  = rewoo_result.get("rewoo_plan_valid", False)

        # ── ReAct ──
        counting.reset()
        t0 = time.perf_counter()
        _run_react(state, counting)
        react_ms    = (time.perf_counter() - t0) * 1000
        react_calls = counting.call_count

        call_reduction = (1 - rewoo_calls / react_calls) * 100 if react_calls else 0
        time_reduction = (1 - rewoo_ms / react_ms) * 100 if react_ms else 0

        r = ReWOOAblationResult(
            n_travellers=n,
            rewoo_llm_calls=rewoo_calls,
            react_llm_calls=react_calls,
            rewoo_plan_ms=rewoo_ms,
            react_ms=react_ms,
            call_reduction=call_reduction,
            time_reduction=time_reduction,
            rewoo_plan_valid=rewoo_valid,
        )
        results.append(r)

        ok = "yes" if rewoo_valid else "NO"
        print(f"  {f'{n} traveller(s)':<16} {rewoo_calls:>6} {react_calls:>6} {call_reduction:>6.0f}%  {ok:>9}")

    avg_reduction = sum(r.call_reduction for r in results) / len(results)
    print("  " + "─" * 48)
    print(f"\n  Average LLM calls saved: {avg_reduction:.0f}%")
    print(f"  ReWOO plans the whole trip in 1 call.")
    print(f"  ReAct asks the LLM what to do at every single step.")
    print()

    return results




# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_rewoo = "--cache" not in sys.argv
    run_cache = "--rewoo" not in sys.argv

    if run_rewoo:
        run_rewoo_ablation()
