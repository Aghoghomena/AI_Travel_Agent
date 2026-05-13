"""
Amadeus flight search tool.
 
Three search modes depending on what information is available:
 
1. SPECIFIC — origin IATA + destination IATA + exact date → flight_offers_search, returns cheapest fare for that leg

2. FLEXIBLE DATE — origin IATA + destination IATA + month only → flight_dates, finds cheapest date within the month then calls flight_offers_search for that date

3. ANYWHERE — origin IATA only (destination = "anywhere") → flight_destinations, returns cheapest destinations from that origin, filtered to candidate destinations if provided
"""

import os
from datetime import datetime
from amadeus import Client, ResponseError
from src.state import FlightResult
from data.iata import IATA_MAP, resolve_iata, MONTH_MAP

def get_amadeus_client() -> Client:
    """Returns an authenticated Amadeus client using env vars."""
    return Client(
        client_id=os.environ["AMADEUS_CLIENT_ID"],
        client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
        hostname=os.environ.get("AMADEUS_HOSTNAME", "test"),
    )

def get_amadeus_client() -> Client:
    """Returns an authenticated Amadeus client using env vars."""
    return Client(
        client_id=os.environ["AMADEUS_CLIENT_ID"],
        client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
        hostname=os.environ.get("AMADEUS_HOSTNAME", "test"),
    )

def resolve_date(travel_month: str | None, outbound_date: str | None) -> tuple[str | None, str | None]:
    """
    Resolves travel time inputs to (exact_date, month_only).
 
    Returns:
        (exact_date, month_only) where:
        - exact_date is YYYY-MM-DD if we have a specific date
        - month_only is YYYY-MM if we only have a month name
        - Both None if no date info available
    """
    # Already have an exact date
    if outbound_date:
        return outbound_date, None
 
    # Have a month name — resolve to mid-month default
    if travel_month:
        month_lower = travel_month.lower().strip()
        month_num = MONTH_MAP.get(month_lower)
        if month_num:
            year = datetime.now().year
            # If month already passed this year, use next year
            if month_num < datetime.now().month:
                year += 1
            mid_day = 15
            return f"{year}-{month_num:02d}-{mid_day}", f"{year}-{month_num:02d}"
 
    return None, None


# ── Search functions ──────────────────────────────────────────

def search_flights_specific(origin_iata: str,destination_iata: str,departure_date: str,adults: int = 1,) -> FlightResult | None:
    """
    Search 1: Specific origin + destination + exact date.
    Returns cheapest one-way fare for that leg.
    """
    client = get_amadeus_client()
 
    try:
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin_iata,
            destinationLocationCode=destination_iata,
            departureDate=departure_date,
            adults=str(adults),
            max=5,
        )
    except ResponseError as e:
        print(f"[amadeus] ResponseError {origin_iata}→{destination_iata} on {departure_date}: {e}")
        return None
 
    offers = response.data
    if not offers:
        return None
 
    cheapest = offers[0]
    price = float(cheapest["price"]["grandTotal"])
    currency = cheapest["price"]["currency"]
 
    return FlightResult(
        origin_iata=origin_iata,
        destination_iata=destination_iata,
        outbound_date=departure_date,
        price_local=price,
        currency=currency,
        price_usd=None,
        fetched_at=datetime.utcnow().isoformat(),
        from_cache=False,
    )

def search_flights_flexible_date(origin_iata: str,destination_iata: str,travel_month: str,adults: int = 1,) -> FlightResult | None:
    """
    Search 2: Specific origin + destination, month only (no exact date).
    Uses flight_dates to find cheapest date in the month, then
    calls search_flights_specific for that date.
    """
    client = get_amadeus_client()
 
    # Resolve month to YYYY-MM
    _, month_str = resolve_date(travel_month, None)
    if not month_str:
        return None
 
    try:
        response = client.shopping.flight_dates.get(
            origin=origin_iata,
            destination=destination_iata,
        )
    except ResponseError as e:
        print(f"[amadeus] flight_dates error {origin_iata}→{destination_iata}: {e}")
        # Fall back to mid-month default
        exact_date, _ = resolve_date(travel_month, None)
        if exact_date:
            return search_flights_specific(origin_iata, destination_iata, exact_date, adults)
        return None
 
    # Filter results to the target month and pick cheapest
    target_month = month_str  # YYYY-MM
    best_date = None
    best_price = float("inf")
 
    for item in response.data:
        dep_date = item.get("departureDate", "")
        if dep_date.startswith(target_month):
            price = float(item.get("price", {}).get("total", float("inf")))
            if price < best_price:
                best_price = price
                best_date = dep_date
 
    if not best_date:
        # No results in that month — fall back to mid-month
        exact_date, _ = resolve_date(travel_month, None)
        if exact_date:
            return search_flights_specific(origin_iata, destination_iata, exact_date, adults)
        return None
 
    return search_flights_specific(origin_iata, destination_iata, best_date, adults)

def search_flights_anywhere_with_specific_dates(origin_iata: str, departure_dates: str, candidate_destinations: list[str] | None = None,) -> list[FlightResult]:
    """
    Mode 3: Origin only, no destination specified.
    Uses flight_destinations to find cheapest reachable destinations.
    If candidate_destinations is provided, filters results to those only.
    Returns a list of FlightResults (one per reachable destination).
    """
    client = get_amadeus_client()
 
    try:
        response = client.shopping.flight_destinations.get(
            origin=origin_iata,
            departureDate=departure_dates,  # Can be YYYY-MM or YYYY-MM-DD
        )
    except ResponseError as e:
        print(f"[amadeus] flight_destinations error from {origin_iata}: {e}")
        return []
 
    results = []
    for item in response.data:
        dest = item.get("destination")
        if not dest:
            continue
 
        # Filter to candidates if provided
        if candidate_destinations and dest not in candidate_destinations:
            continue
 
        price = float(item.get("price", {}).get("total", 0))
        currency = "USD"  # flight_destinations returns USD
        dep_date = item.get("departureDate", datetime.utcnow().strftime("%Y-%m-%d"))
 
        results.append(FlightResult(
            origin_iata=origin_iata,
            destination_iata=dest,
            outbound_date=departure_dates,
            price_local=price,
            currency=currency,
            price_usd=price,   # already USD from this endpoint
            fetched_at=datetime.utcnow().isoformat(),
            from_cache=False,
        ))
 
    return results

# ── Unified entry point ───────────────────────────────────────
 
def search_flights(origin_iata: str,destination_iata: str | None,travel_month: str | None = None,outbound_date: str | None = None,candidate_destinations: list[str] | None = None,adults: int = 1,) -> list[FlightResult]:
    """
    Unified flight search. Picks the right Amadeus endpoint based
    on what information is available.
 
    Args:
        origin_iata: Departure airport IATA code.
        destination_iata: Destination IATA code, or None for anywhere.
        travel_month: Month name e.g. "June" — used if no exact date.
        outbound_date: Exact date YYYY-MM-DD — takes priority over month.
        candidate_destinations: Filter for anywhere search.
        adults: Number of passengers.
 
    Returns:
        List of FlightResult objects (empty list on failure).
    """
    # Mode 3 — anywhere
    if not destination_iata or destination_iata.lower() == "anywhere":
        return search_flights_anywhere(origin_iata, candidate_destinations)
 
    # Resolve date
    exact_date, _ = resolve_date(travel_month, outbound_date)
 
    # Mode 1 — specific date
    if exact_date:
        result = search_flights_specific(origin_iata, destination_iata, exact_date, adults)
        return [result] if result else []
 
    # Mode 2 — month only, no exact date
    if travel_month:
        result = search_flights_flexible_date(origin_iata, destination_iata, travel_month, adults)
        return [result] if result else []
 
    print(f"[amadeus] No date or month provided for {origin_iata}→{destination_iata}")
    return []
    """
    Searches for the cheapest one-way flight on a given leg.
 
    Args:
        origin_iata: 3-letter IATA code for origin airport.
        destination_iata: 3-letter IATA code for destination airport.
        departure_date: Date in YYYY-MM-DD format.
        adults: Number of adult passengers (default 1).
 
    Returns:
        FlightResult with the cheapest fare found, or None if no
        results or an API error occurs.
    """
    client = get_amadeus_client()
 
    try:
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin_iata,
            destinationLocationCode=destination_iata,
            departureDate=departure_date,
            adults=str(adults),
            max=5,
        )
    except ResponseError as e:
        print(f"[amadeus] ResponseError for {origin_iata}→{destination_iata}: {e}")
        return None
 
    offers = response.data
    if not offers:
        return None
 
    # Offers are already sorted cheapest first by the API
    cheapest = offers[0]
    price = float(cheapest["price"]["grandTotal"])
    currency = cheapest["price"]["currency"]
 
    return FlightResult(
        origin_iata=origin_iata,
        destination_iata=destination_iata,
        price_local=price,
        currency=currency,
        price_usd=None,          # normalised by currency agent (Step 8)
        fetched_at=datetime.utcnow().isoformat(),
        from_cache=False,
    )