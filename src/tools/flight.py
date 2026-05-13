

"""
Sky Scrapper (RapidAPI) flight search tool.
 
Two-step process per route:
  1. Resolve IATA → skyId + entityId (static map first, API fallback)
  2. Call searchFlights with origin/destination skyId + entityId + date
 
Requires in .env:
  RAPIDAPI_KEY=your_key
  RAPIDAPI_HOST=sky-scrapper.p.rapidapi.com
"""
 
import os
import httpx
from datetime import datetime
from src.state import FlightResult
from data.iata import IATA_MAP, MONTH_MAP, AIRPORT_MAP
 
RAPIDAPI_HOST = "sky-scrapper.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}/api/v1/flights"
TIMEOUT = 15.0
 
def _headers() -> dict:
    return {
        "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }


 
def resolve_date(travel_month: str | None, outbound_date: str | None) -> str | None:
    """
    Resolves travel time to a YYYY-MM-DD date string.
    Exact date takes priority. Month name falls back to mid-month.
    Returns None if neither is provided.
    """
    if outbound_date:
        return outbound_date
 
    if travel_month:
        month_num = MONTH_MAP.get(travel_month.lower().strip())
        if month_num:
            year = datetime.now().year
            if month_num < datetime.now().month:
                year += 1
            return f"{year}-{month_num:02d}-15"
 
    return None


def lookup_airport(iata: str) -> dict | None:
    """
    Returns skyId + entityId for an IATA code.
    Checks static map first, then calls searchAirport API.
    """
    # Static map hit
    if iata in AIRPORT_MAP:
        return AIRPORT_MAP[iata]
 
    # API fallback
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/searchAirport",
                headers=_headers(),
                params={"query": iata, "locale": "en-US"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            if data:
                airport = data[0]
                result = {
                    "skyId": airport.get("skyId", iata),
                    "entityId": airport.get("entityId", ""),
                    "name": airport.get("presentation", {}).get("title", iata),
                }
                # Cache in map for this session
                AIRPORT_MAP[iata] = result
                return result
    except Exception as e:
        print(f"[flights] searchAirport error for {iata}: {e}")
 
    return None


def search_flights(origin_iata: str,destination_iata: str,travel_month: str | None = None,outbound_date: str | None = None,adults: int = 1,) -> FlightResult | None:
    """
    Searches for the cheapest one-way flight on a specific leg.
 
    Args:
        origin_iata: Departure airport IATA code.
        destination_iata: Destination airport IATA code.
        travel_month: Month name e.g. "June" — used if no exact date.
        outbound_date: Exact date YYYY-MM-DD — takes priority over month.
        adults: Number of passengers.
 
    Returns:
        FlightResult with cheapest fare, or None on failure.
    """
    departure_date = resolve_date(travel_month, outbound_date)
    if not departure_date:
        print(f"[flights] No date resolved for {origin_iata}→{destination_iata}")
        return None
 
    origin = lookup_airport(origin_iata)
    destination = lookup_airport(destination_iata)
 
    if not origin:
        print(f"[flights] Could not resolve airport: {origin_iata}")
        return None
    if not destination:
        print(f"[flights] Could not resolve airport: {destination_iata}")
        return None
 
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/searchFlights",
                headers=_headers(),
                params={
                    "originSkyId": origin["skyId"],
                    "destinationSkyId": destination["skyId"],
                    "originEntityId": origin["entityId"],
                    "destinationEntityId": destination["entityId"],
                    "date": departure_date,
                    "adults": str(adults),
                    "currency": "USD",
                    "market": "en-US",
                    "countryCode": "US",
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        print(f"[flights] searchFlights error {origin_iata}→{destination_iata}: {e}")
        return None
 
    # Parse cheapest itinerary
    itineraries = (
        data.get("data", {})
            .get("itineraries", [])
    )
    if not itineraries:
        print(f"[flights] No itineraries found: {origin_iata}→{destination_iata} on {departure_date}")
        return None
 
    # Already sorted cheapest first by Sky Scrapper
    cheapest = itineraries[0]
    price = float(cheapest.get("price", {}).get("raw", 0))
 
    return FlightResult(
        origin_iata=origin_iata,
        destination_iata=destination_iata,
        price_local=price,
        currency="USD",   # Sky Scrapper returns USD
        price_usd=price,  # already USD — no currency conversion needed
        fetched_at=datetime.utcnow().isoformat(),
        from_cache=False,
    )
 