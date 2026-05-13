# ── IATA lookup map ───────────────────────────────────────────
IATA_MAP = {
    "dublin": "DUB",
    "lagos": "LOS",
    "toronto": "YYZ",
    "istanbul": "IST",
    "lisbon": "LIS",
    "dubai": "DXB",
    "amsterdam": "AMS",
    "london": "LHR",
    "tokyo": "NRT",
    "sydney": "SYD",
    "new york": "JFK",
    "accra": "ACC",
    "nairobi": "NBO",
    "singapore": "SIN",
    "paris": "CDG",
    "rome": "FCO",
    "berlin": "BER",
    "montreal": "YUL",
    "vancouver": "YVR",
}

# Month name → number
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def resolve_iata(city_name: str) -> str | None:
    """Resolves a city name to IATA code. Returns None if unknown."""
    return IATA_MAP.get(city_name.lower().strip())

REGION_AIRPORTS = {
    "Africa": [
        {
            "iata": "RAK",
            "city": "Marrakech",
            "country": "Morocco",
            "priority": 10
        },
        {
            "iata": "CMN",
            "city": "Casablanca",
            "country": "Morocco",
            "priority": 9
        },
        {
            "iata": "CAI",
            "city": "Cairo",
            "country": "Egypt",
            "priority": 10
        },
        {
            "iata": "TUN",
            "city": "Tunis",
            "country": "Tunisia",
            "priority": 8
        },
        {
            "iata": "NBO",
            "city": "Nairobi",
            "country": "Kenya",
            "priority": 7
        },
        {
            "iata": "CPT",
            "city": "Cape Town",
            "country": "South Africa",
            "priority": 8
        },
    ],
    "Europe":    [
    {
        "iata": "LIS",
        "city": "Lisbon",
        "country": "Portugal",
        "priority": 10
    },
    {
        "iata": "MAD",
        "city": "Madrid",
        "country": "Spain",
        "priority": 10
    },
    {
        "iata": "BCN",
        "city": "Barcelona",
        "country": "Spain",
        "priority": 10
    },
    {
        "iata": "CDG",
        "city": "Paris",
        "country": "France",
        "priority": 10
    },
    {
        "iata": "FCO",
        "city": "Rome",
        "country": "Italy",
        "priority": 9
    },
    {
        "iata": "ATH",
        "city": "Athens",
        "country": "Greece",
        "priority": 8
    },
]
}

AIRPORT_MAP: dict[str, dict] = {
    "DUB": {"skyId": "DUB", "entityId": "95673375", "name": "Dublin"},
    "LOS": {"skyId": "LOS", "entityId": "95673483", "name": "Lagos"},
    "YYZ": {"skyId": "YYZ", "entityId": "95673834", "name": "Toronto"},
    "IST": {"skyId": "IST", "entityId": "95673633", "name": "Istanbul"},
    "LIS": {"skyId": "LIS", "entityId": "95673477", "name": "Lisbon"},
    "DXB": {"skyId": "DXB", "entityId": "95673506", "name": "Dubai"},
    "AMS": {"skyId": "AMS", "entityId": "95673352", "name": "Amsterdam"},
    "LHR": {"skyId": "LHR", "entityId": "95673529", "name": "London"},
    "NRT": {"skyId": "NRT", "entityId": "95673699", "name": "Tokyo"},
    "SYD": {"skyId": "SYD", "entityId": "95673781", "name": "Sydney"},
    "JFK": {"skyId": "JFK", "entityId": "95673641", "name": "New York"},
    "ACC": {"skyId": "ACC", "entityId": "95673340", "name": "Accra"},
    "NBO": {"skyId": "NBO", "entityId": "95673561", "name": "Nairobi"},
    "SIN": {"skyId": "SIN", "entityId": "95673762", "name": "Singapore"},
    "CDG": {"skyId": "CDG", "entityId": "95673410", "name": "Paris"},
    "FCO": {"skyId": "FCO", "entityId": "95673426", "name": "Rome"},
    "BER": {"skyId": "BER", "entityId": "95673374", "name": "Berlin"},
    "YUL": {"skyId": "YUL", "entityId": "95673833", "name": "Montreal"},
    "YVR": {"skyId": "YVR", "entityId": "95673835", "name": "Vancouver"},
}

def get_airports_by_region(region):
    return REGION_AIRPORTS.get(region, [])

MOCK_HOTEL_PRICES: dict[str, float] = {
    # ── Europe ────────────────────────────────────────────────
    "DUB": 130.0,   # Dublin
    "LHR": 175.0,   # London
    "AMS": 160.0,   # Amsterdam
    "LIS": 110.0,   # Lisbon
    "IST": 90.0,    # Istanbul — affordable, lira weakness
    "CDG": 165.0,   # Paris
    "FCO": 140.0,   # Rome
    "BER": 120.0,   # Berlin
    "MAD": 135.0,   # Madrid
    "BCN": 140.0,   # Barcelona
    "MXP": 145.0,   # Milan
    "VIE": 150.0,   # Vienna
    "ATH": 115.0,   # Athens
    "CPH": 170.0,   # Copenhagen
    "ARN": 165.0,   # Stockholm
 
    # ── Africa ────────────────────────────────────────────────
    "LOS": 120.0,   # Lagos — expensive due to limited supply
    "ACC": 108.0,   # Accra — Ghana avg ~$108/night
    "NBO": 95.0,    # Nairobi
    "CAI": 75.0,    # Cairo
    "CMN": 85.0,    # Casablanca
    "DAK": 90.0,    # Dakar
    "ADD": 140.0,   # Addis Ababa — most expensive in Africa
    "CPT": 100.0,   # Cape Town
    "JNB": 80.0,    # Johannesburg
    "DAR": 79.0,    # Dar es Salaam
    "KGL": 85.0,    # Kigali
    "EBB": 90.0,    # Kampala (Entebbe airport)
    "ABV": 110.0,   # Abuja
    "DKR": 90.0,    # Dakar (Léopold Sédar Senghor)
    "TUN": 70.0,    # Tunis
    "RAK": 80.0,    # Marrakech
    "ALG": 95.0,    # Algiers
}
 
 # ── IATA → currency mapping (Africa + Europe MVP scope) ───────
IATA_TO_CURRENCY: dict[str, str] = {
    # Europe
    "DUB": "EUR",
    "LHR": "GBP",
    "AMS": "EUR",
    "LIS": "EUR",
    "IST": "TRY",
    "CDG": "EUR",
    "FCO": "EUR",
    "BER": "EUR",
    "MAD": "EUR",
    "BCN": "EUR",
    "MXP": "EUR",
    "VIE": "EUR",
    "ATH": "EUR",
    "CPH": "DKK",
    "ARN": "SEK",
    # Africa
    "LOS": "NGN",
    "ABV": "NGN",
    "ACC": "GHS",
    "NBO": "KES",
    "CAI": "EGP",
    "CMN": "MAD",
    "RAK": "MAD",
    "DKR": "XOF",
    "DAK": "XOF",
    "ADD": "ETB",
    "CPT": "ZAR",
    "JNB": "ZAR",
    "DAR": "TZS",
    "KGL": "RWF",
    "EBB": "UGX",
    "TUN": "TND",
    "ALG": "DZD",
}

# ── City centre coordinates for all MVP hub cities ────────────
CITY_CENTRES: dict[str, tuple[float, float]] = {
    # Europe
    "DUB": (53.3498,  -6.2603),   # Dublin
    "LHR": (51.5074,  -0.1278),   # London
    "AMS": (52.3676,   4.9041),   # Amsterdam
    "LIS": (38.7169,  -9.1399),   # Lisbon
    "IST": (41.0082,  28.9784),   # Istanbul
    "CDG": (48.8566,   2.3522),   # Paris
    "FCO": (41.9028,  12.4964),   # Rome
    "BER": (52.5200,  13.4050),   # Berlin
    "CPH": (55.6761,  12.5683),   # Copenhagen
    "ARN": (59.3293,  18.0686),   # Stockholm
    # Africa
    "LOS": ( 6.5244,   3.3792),   # Lagos
    "ABV": ( 9.0579,   7.4951),   # Abuja
    "ACC": ( 5.6037,  -0.1870),   # Accra
    "NBO": (-1.2921,  36.8219),   # Nairobi
    "CAI": (30.0444,  31.2357),   # Cairo
    "CMN": (33.5731,  -7.5898),   # Casablanca
    "RAK": (31.6295,  -7.9811),   # Marrakech
    "DKR": (14.7167, -17.4677),   # Dakar
    "ADD": ( 9.0320,  38.7469),   # Addis Ababa
    "CPT": (-33.9249,  18.4241),  # Cape Town
    "JNB": (-26.2041,  28.0473),  # Johannesburg
    "DAR": (-6.7924,  39.2083),   # Dar es Salaam
    "KGL": (-1.9441,  30.0619),   # Kigali
    "EBB": ( 0.3136,  32.5811),   # Kampala
    "TUN": (36.8065,  10.1815),   # Tunis
    "ALG": (36.7372,   3.0865),   # Algiers
}

# Activity types to search for
INCLUDED_TYPES = [
    "tourist_attraction",
    "museum",
    "art_gallery",
    "park",
    "historical_landmark",
]
 