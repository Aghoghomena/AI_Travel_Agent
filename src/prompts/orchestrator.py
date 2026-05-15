"""
Orchestrator Agent prompt.
 
Generates a ReWOO XML execution plan before any API calls fire.
The plan lists every tool call needed to evaluate all candidate
destinations for all travellers.
"""

ORCHESTRATOR_PROMPT = """
You are the orchestrator for a group travel planning system. Your job is to generate a complete XML execution plan for finding
the cheapest destination for a group of travellers to meet.
 
You will be given:
- A list of travellers with their origin IATA codes
- A list of candidate destination IATA codes
- Travel month and duration
- Search mode: how the destinations were chosen

─────────────────────────────────────────
SEARCH MODE
─────────────────────────────────────────
- "specific"  — user requested these exact destinations; search ALL of them regardless of flight availability
- "region"    — destinations are semantic matches for a region; include only those reachable by all travellers
- "anywhere"  — destinations are general hubs; include only those reachable by all travellers

─────────────────────────────────────────
PLAN RULES
─────────────────────────────────────────
1. Create one traveller_agent step per traveller
2. Create one accommodation_agent step for all destinations
3. Create one currency_agent step after flights + accommodation
4. Create one ranker_agent step after currency
5. Steps that can run in parallel have no depends_on or same depends_on
6. Traveller steps are always parallel (no depends_on)
7. Accommodation step is parallel to traveller steps (no depends_on)
8. Currency step depends_on ALL traveller step ids + accommodation step id
9. Ranker step depends_on currency step id
10. Return ONLY the XML, no prose, no markdown backticks
 
─────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────
 
<plan>
  <step id="1" tool="traveller_agent" traveller="Traveller 1"
        origin="DUB" destinations="IST,LIS,AMS" />
  <step id="2" tool="traveller_agent" traveller="Traveller 2"
        origin="LOS" destinations="IST,LIS,AMS" />
  <step id="3" tool="accommodation_agent"
        destinations="IST,LIS,AMS" />
  <step id="4" tool="currency_agent" depends_on="1,2,3" />
  <step id="5" tool="ranker_agent" depends_on="4" />
</plan>
 

"""

 
def build_orchestrator_prompt(
    travellers: list,
    candidate_destinations: list[str],
    travel_month: str | None,
    duration_nights: int | None,
    search_mode="anywhere"
) -> str:
    traveller_lines = "\n".join(
        f"  - {t.name} ({t.origin_iata})" for t in travellers
    )
    dest_str = ", ".join(candidate_destinations)
 
    context = (
        f"Search Mode: {search_mode}\n"
        f"Travellers:\n{traveller_lines}\n"
        f"Destinations: {dest_str}\n"
        f"Month: {travel_month}\n"
        f"Nights: {duration_nights}\n"
    )
 
    return ORCHESTRATOR_PROMPT + context + "\nOutput:"