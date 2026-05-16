"""
Amadeus flight search tool.
 
Three search modes depending on what information is available:
 
1. SPECIFIC — origin IATA + destination IATA + exact date → flight_offers_search, returns cheapest fare for that leg

2. FLEXIBLE DATE — origin IATA + destination IATA + month only → flight_dates, finds cheapest date within the month then calls flight_offers_search for that date

3. ANYWHERE — origin IATA only (destination = "anywhere") → flight_destinations, returns cheapest destinations from that origin, filtered to candidate destinations if provided
"""

import os
from datetime import datetime, timedelta
from src.state import FlightResult
from data.iata import IATA_MAP, resolve_iata, MONTH_MAP, REGIONS, COUNTRY_TO_CONTINENT
import requests
import http.client
import json
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SEARCH_API_KEY")
API_KEY2 = os.getenv("GOOGLE_KNOWLEDGE_API_KEY")
BASE_URL = "https://www.searchapi.io/api/v1/search"


def normalize_destination(data, origin_iata: str, region: str | None = None, destination_country: list[str] | None = None):
    results = []

    destinations = data.get("destinations", [])

    region_continent = region.title() if region else None

    for destination in destinations:

        if not destination.get("flight"):
            continue

        if region_continent and COUNTRY_TO_CONTINENT.get(destination.get("country")) != region_continent:
            continue

        if destination_country and destination.get("country") not in destination_country:
            continue

         # collect both dates
        dates = [destination.get("outbound_date")]

        if destination.get("alternative_outbound_date"):
            dates.append(destination.get("alternative_outbound_date"))


        results.append({
            "origin_iata": origin_iata,
            "country": destination["country"],
            "destination_iata": destination["primary_airport"],
            "price_usd": destination["flight"]["price"],
            "avg_cost_per_night": destination.get("avg_cost_per_night", 0),
            "Outbound_flights": dates,
            "fetched_at": datetime.now()

        })

    return results


def resolve_start_date(travel_month: str | None = None, outbound_date: str | None = None) -> str:
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
        return outbound_date
 
    # Have a month name — resolve to mid-month default
    elif travel_month:
        month_lower = travel_month.lower().strip()
        month_num = MONTH_MAP.get(month_lower)
        if month_num:
            year = datetime.now().year
            # If month already passed this year, use next year
            if month_num < datetime.now().month:
                year += 1
            start_date = "01"
            return f"{year}-{month_num:02d}-{start_date}"
    
    else:
        return datetime.now().strftime("%Y-%m-%d")
 

def resolve_end_date(travel_month: str | None = None, outbound_date: str | None = None, duration_nights: str | None = None) -> str:
    """
    Resolves travel time inputs to (exact_date, month_only).
 
    Returns:
        (exact_date, month_only) where:
        - exact_date is YYYY-MM-DD if we have a specific date
        - month_only is YYYY-MM if we only have a month name
        - Both None if no date info available
    """
    if outbound_date and duration_nights:
        try:
            start = datetime.strptime(outbound_date, "%Y-%m-%d")
            end = start + timedelta(days=int(duration_nights))
            return end.strftime("%Y-%m-%d")
        except ValueError:
            pass

    if travel_month:
        month_lower = travel_month.lower().strip()
        month_num = MONTH_MAP.get(month_lower)
        if month_num:
            year = datetime.now().year
            if month_num < datetime.now().month:
                year += 1
            return f"{year}-{month_num:02d}-27"



def search_everywhere_with_specific_date(origin_iata: str,outbound_date: str, durations_nights: str, adults: int = 1, ):
    """
    Endpoint 1: Cheapest flights for a whole month using the Calendar API a week trip in the month.
    origin / destination: IATA airport codes e.g. "DUB", "LHR"
    """

    start_date = resolve_start_date("", outbound_date)

    end_date = resolve_end_date("", outbound_date, durations_nights)

    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "time_period": f"{start_date}..{end_date}",
        "api_key": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata)
 
    return normalized

def search_everywhere_by_month(origin_iata: str,travel_month: str, adults: int = 1, ):
    """
    Endpoint 1: Cheapest flights for a whole month using the Calendar API a week trip in the month and a specific location.
    origin / destination: IATA airport codes e.g. "DUB", "LHR"
    """
    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "time_period": f"one_week_trip_in_{travel_month}",   
        "currency": "USD",
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)

 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata)
 
    return normalized

def search_region_by_month(origin_iata: str,travel_month: str, travel_region: str, adults: int = 1, ):
    """
    Endpoint 1: Cheapest flights for a whole month using the Calendar API a week trip in the month and a specific location.
    origin / destination: IATA airport codes e.g. "DUB", "LHR" filter by region
    """
    region_kgmid = REGIONS.get(travel_region.lower())
    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "arrival_id": region_kgmid,
        "time_period": f"one_week_trip_in_{travel_month}",   
        "currency": "USD",
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)

 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata, travel_region)
 
    return normalized

def search_region_with_specific_date(origin_iata: str,outbound_date: str, durations_nights: str, travel_region:str, adults: int = 1, ):
    """
    Endpoint 4: Cheapest flights for a whole month using the Calendar API a week trip in the month.
    origin / destination: IATA airport codes e.g. "DUB", "LHR"
    """
    start_date = resolve_start_date("", outbound_date)
    end_date = resolve_end_date("", outbound_date, durations_nights)
    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "time_period": f"{start_date}..{end_date}",
        "api_key": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata, travel_region)
 
    return normalized


def search_location_by_month(origin_iata: str, travel_month: str, destination_country: list[str], adults: int = 1, ):
    """
    Endpoint 1: Cheapest flights for a whole month using the Calendar API a week trip in the month and a specific location.
    origin / destination: IATA airport codes e.g. "DUB", "LHR" filter by region
    """

    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "time_period": f"one_week_trip_in_{travel_month}",   
        "currency": "USD",
        "api_key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)

 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata, "", destination_country)
 
    return normalized

def search_location_specific_date(origin_iata: str, outbound_date: str, durations_nights: str, destination_country: list[str], adults: int = 1, ):
    """
    Endpoint 4: Cheapest flights for a whole month using the Calendar API a week trip in the month.
    origin / destination: IATA airport codes e.g. "DUB", "LHR"
    """

    start_date = resolve_start_date("", outbound_date)

    end_date = resolve_end_date("", outbound_date, durations_nights)

    params = {
        "engine": "google_travel_explore",
        "departure_id": origin_iata,
        "time_period": f"{start_date}..{end_date}",
        "api_key": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
 
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return
 
    data = response.json()
    normalized = normalize_destination(data, origin_iata, "", destination_country)
 
    return normalized


def search_flights(
    origin_iata: str,
    travel_month: str | None = None,
    outbound_date: str | None = None,
    duration_nights: str | None = None,
    travel_region: str | None = None,
    destination_country: list[str] | None = None,
):
    """
    Routes to the correct search function based on available inputs.

    Priority: region > country > anywhere
    Date mode: specific date > month
    """
    if travel_region and outbound_date:
        return search_region_with_specific_date(origin_iata, outbound_date, duration_nights, travel_region)

    if travel_region and travel_month:
        return search_region_by_month(origin_iata, travel_month, travel_region)

    if destination_country and outbound_date:
        return search_location_specific_date(origin_iata, outbound_date, duration_nights, destination_country)

    if destination_country and travel_month:
        return search_location_by_month(origin_iata, travel_month, destination_country)

    if outbound_date:
        return search_everywhere_with_specific_date(origin_iata, outbound_date, duration_nights)

    if travel_month:
        return search_everywhere_by_month(origin_iata, travel_month)

    raise ValueError("Provide at least travel_month or outbound_date")
