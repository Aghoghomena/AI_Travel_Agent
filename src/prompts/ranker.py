"""
Ranker Agent prompt.
 
Used for chain-of-thought ranking to catch arithmetic errors.
"""
RANKER_PROMPT = """
You are a cost ranker for a group travel planning system.
 
You will be given a list of candidate destinations with their 
total costs in USD. Your job is to rank them cheapest first.
 
─────────────────────────────────────────
RULES
─────────────────────────────────────────
1. Show your working — list each destination and its grand_total_usd
2. Rank from cheapest (rank 1) to most expensive
3. Return ONLY a JSON array, no prose, no markdown backticks
 
─────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────
 
[
  {"iata": "IST", "grand_total_usd": 1240.0, "rank": 1},
  {"iata": "LIS", "grand_total_usd": 1480.0, "rank": 2},
  {"iata": "AMS", "grand_total_usd": 1720.0, "rank": 3}
]
 
─────────────────────────────────────────
EXAMPLE
─────────────────────────────────────────
 
Destinations:
- AMS: flights=$980 + accommodation=$480 = grand_total=$1460
- IST: flights=$640 + accommodation=$270 = grand_total=$910
- LIS: flights=$720 + accommodation=$330 = grand_total=$1050
 
Working:
IST $910 → rank 1
LIS $1050 → rank 2
AMS $1460 → rank 3
 
Output: [{"iata": "IST", "grand_total_usd": 910.0, "rank": 1}, {"iata": "LIS", "grand_total_usd": 1050.0, "rank": 2}, {"iata": "AMS", "grand_total_usd": 1460.0, "rank": 3}]
 
"""

def build_ranker_prompt(destination_results: list) -> str:
    lines = ["Destinations:"]
    for d in destination_results:
        flights = d.total_flight_cost_usd or 0.0
        accom = d.total_accommodation_usd or 0.0
        total = d.grand_total_usd or (flights + accom)
        lines.append(
            f"- {d.iata}: flights=${flights} + accommodation=${accom} = grand_total=${total}"
        )
    return RANKER_PROMPT + "\n".join(lines) + "\nOutput:"