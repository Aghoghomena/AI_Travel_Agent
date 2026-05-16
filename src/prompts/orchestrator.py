"""
Orchestrator Agent prompt.

Generates a ReWOO XML execution plan before any API calls fire.
The plan covers the full lifecycle: memory check, search, rank, persist.
"""

ORCHESTRATOR_PROMPT = """
You are the orchestrator for a group travel planning system using the ReWOO pattern.
Your job is to generate an XML execution plan that decides which tools to call,
in what order, and with what parameters — before any tool runs.

You will be given travellers, candidate destinations, travel details, and user constraints.
Use the constraints to decide which optional steps to include or skip.

─────────────────────────────────────────
AVAILABLE TOOLS
─────────────────────────────────────────
REQUIRED:
- memory_read        — Check episodic cache for prior results. Always step 1.
- traveller_agent    — Search flights for ONE traveller. One step per traveller.
- currency_agent     — Convert all prices to USD. Depends on all traveller steps (+ accommodation if included).
- ranker_agent       — Rank destinations by total cost. Depends on currency step.
- memory_write       — Persist results. Always last step.

OPTIONAL — include only when the constraint applies:
- accommodation_agent — Fetch hotel costs. Include UNLESS accommodation_needed=false.
  One step for all destinations, depends_on memory_read, runs parallel to traveller steps.
- direct_flights_filter — Remove destinations with no direct flight for any traveller.
  Include ONLY if direct_flights_only=true. Depends on all traveller steps, runs before currency_agent.
- budget_filter      — Remove destinations over the per-person budget.
  Include ONLY if max_budget_usd is set. Add after ranker_agent, before memory_write.
  Attribute: max_budget_usd="<value>"

─────────────────────────────────────────
DEPENDENCY RULES
─────────────────────────────────────────
1. memory_read: no depends_on
2. traveller_agent steps: each depends_on memory_read step id
3. accommodation_agent (if included): depends_on memory_read step id, parallel with traveller steps
4. direct_flights_filter (if included): depends_on ALL traveller step ids
5. currency_agent: depends_on ALL traveller step ids + accommodation step id (if included) + direct_flights_filter step id (if included)
6. ranker_agent: depends_on currency step id
7. budget_filter (if included): depends_on ranker step id
8. memory_write: depends_on ranker step id if no budget_filter, otherwise depends_on budget_filter step id

Return ONLY the XML — no prose, no markdown.

─────────────────────────────────────────
EXAMPLES
─────────────────────────────────────────

Standard search (no special constraints):
<plan>
  <step id="1" tool="memory_read" />
  <step id="2" tool="traveller_agent" traveller="Alice" origin="DUB" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="3" tool="traveller_agent" traveller="Bob" origin="LOS" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="4" tool="accommodation_agent" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="5" tool="currency_agent" depends_on="2,3,4" />
  <step id="6" tool="ranker_agent" depends_on="5" />
  <step id="7" tool="memory_write" depends_on="6" />
</plan>

With budget $400/person and direct flights only, no accommodation:
<plan>
  <step id="1" tool="memory_read" />
  <step id="2" tool="traveller_agent" traveller="Alice" origin="DUB" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="3" tool="traveller_agent" traveller="Bob" origin="LOS" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="4" tool="direct_flights_filter" depends_on="2,3" />
  <step id="5" tool="currency_agent" depends_on="4" />
  <step id="6" tool="ranker_agent" depends_on="5" />
  <step id="7" tool="budget_filter" max_budget_usd="400" depends_on="6" />
  <step id="8" tool="memory_write" depends_on="7" />
</plan>

"""


def build_replan_prompt(current_plan: str, user_feedback: str) -> str:
    return (
        ORCHESTRATOR_PROMPT
        + f"CURRENT PLAN:\n{current_plan}\n\n"
        f"USER FEEDBACK (you MUST apply this exactly — do not ignore any instruction):\n"
        f"  {user_feedback}\n\n"
        f"Output the updated plan XML incorporating the feedback. "
        f"Keep all unchanged steps identical.\nOutput:"
    )


def build_orchestrator_prompt(
    travellers: list,
    candidate_destinations: list[str],
    travel_month: str | None,
    duration_nights: int | None,
    search_mode: str = "anywhere",
    accommodation_needed: bool = True,
    max_budget_usd: float | None = None,
    direct_flights_only: bool = False,
    user_feedback: str | None = None,
) -> str:
    traveller_lines = "\n".join(
        f"  - {t.name} ({t.origin_iata})" for t in travellers
    )
    dest_str = ", ".join(candidate_destinations)

    constraints = []
    constraints.append(f"accommodation_needed={accommodation_needed}")
    constraints.append(f"max_budget_usd={max_budget_usd if max_budget_usd else 'none'}")
    constraints.append(f"direct_flights_only={direct_flights_only}")

    context = (
        f"Search Mode: {search_mode}\n"
        f"Travellers:\n{traveller_lines}\n"
        f"Destinations: {dest_str}\n"
        f"Month: {travel_month}\n"
        f"Nights: {duration_nights}\n"
        f"Constraints: {', '.join(constraints)}\n"
    )

    if user_feedback:
        context += (
            f"\nUSER FEEDBACK ON PREVIOUS PLAN (you must follow this):\n"
            f"  {user_feedback}\n"
        )

    return ORCHESTRATOR_PROMPT + context + "\nOutput:"
