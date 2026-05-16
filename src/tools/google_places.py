"""
Google Places tool.
 
Uses Places API (New) — POST to searchNearby.
Returns top 3 tourist attractions within 15km of city centre,
sorted by rating descending.
 
Requires in .env:
  GOOGLE_PLACES_API_KEY=your_key
"""

import os
import httpx
from src.state import Activity
from data.iata import CITY_CENTRES, INCLUDED_TYPES

PLACES_URL = "https://places.googleapis.com/v1/places:searchNearby"
TIMEOUT = 15.0
RADIUS_METRES = 15000
MAX_RESULTS = 10       # fetch more, then pick top 3 by rating
TOP_N = 3



def get_activities(destination_iata: str) -> list[Activity]:
    """
    Fetches top 3 tourist activities near city centre.
    Returns list of Activity objects sorted by rating (highest first).
    Returns empty list on API error.
    """
    coords = CITY_CENTRES.get(destination_iata.upper())
    if not coords:
        print(f"[places] No coordinates for {destination_iata} so use Paris")
        coords = (48.8566, 2.3522)  # Paris as geographic centre fallback
 
    lat, lng = coords
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "")
 
    payload = {
        "includedTypes": INCLUDED_TYPES,
        "maxResultCount": MAX_RESULTS,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(RADIUS_METRES),
            }
        },
        "rankPreference": "POPULARITY",
    }
 
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,"
            "places.types,"
            "places.rating,"
            "places.userRatingCount,"
            "places.formattedAddress,"
            "places.location,"
            "places.priceLevel"
        ),
    }
 
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(PLACES_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        print(f"[places] API error for {destination_iata}: {e}")
        return []
 
    places = data.get("places", [])
    activities = []
 
    for place in places:
        name = place.get("displayName", {}).get("text", "Unknown")
        rating = place.get("rating")
        address = place.get("formattedAddress", "")
        price_level = place.get("priceLevel", "")
        location = place.get("location", {})
 
        # Estimate distance from city centre
        place_lat = location.get("latitude", lat)
        place_lng = location.get("longitude", lng)
        distance_km = _haversine_km(lat, lng, place_lat, place_lng)
 
        # Estimate cost from price level
        cost_usd, is_free = _estimate_cost(price_level)
 
        # Category from first type
        types = place.get("types", [])
        category = types[0].replace("_", " ").title() if types else "Attraction"
 
        activities.append(Activity(
            name=name,
            category=category,
            estimated_cost_usd=cost_usd,
            distance_km=round(distance_km, 1),
            rating=rating,
            address=address,
            is_free=is_free,
        ))
 
    # Sort by rating descending, return top 3
    activities.sort(key=lambda a: a.rating or 0, reverse=True)
    return activities[:TOP_N]

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance in km between two lat/lng points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def _estimate_cost(price_level: str) -> tuple[float | None, bool]:
    """
    Maps Google price level to estimated USD cost and is_free flag.
    PRICE_LEVEL_FREE / empty → free
    PRICE_LEVEL_INEXPENSIVE → $10
    PRICE_LEVEL_MODERATE → $25
    PRICE_LEVEL_EXPENSIVE → $50
    PRICE_LEVEL_VERY_EXPENSIVE → $100
    """
    mapping = {
        "PRICE_LEVEL_FREE":           (0.0,   True),
        "PRICE_LEVEL_INEXPENSIVE":    (10.0,  False),
        "PRICE_LEVEL_MODERATE":       (25.0,  False),
        "PRICE_LEVEL_EXPENSIVE":      (50.0,  False),
        "PRICE_LEVEL_VERY_EXPENSIVE": (100.0, False),
    }
    return mapping.get(price_level, (0.0, True))