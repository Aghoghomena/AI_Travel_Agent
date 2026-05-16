# ── IATA lookup map ───────────────────────────────────────────
IATA_MAP = {
    # ── Europe ────────────────────────────────────────────────
    "dublin": "DUB",
    "amsterdam": "AMS",
    "london": "LHR",
    "london heathrow": "LHR",
    "london gatwick": "LGW",
    "paris": "CDG",
    "paris charles de gaulle": "CDG",
    "rome": "FCO",
    "rome fiumicino": "FCO",
    "berlin": "BER",
    "madrid": "MAD",
    "barcelona": "BCN",
    "lisbon": "LIS",
    "istanbul": "IST",
    "athens": "ATH",
    "vienna": "VIE",
    "milan": "MXP",
    "milan malpensa": "MXP",
    "copenhagen": "CPH",
    "stockholm": "ARN",
    "oslo": "OSL",
    "helsinki": "HEL",
    "brussels": "BRU",
    "zurich": "ZRH",
    "geneva": "GVA",
    "prague": "PRG",
    "budapest": "BUD",
    "warsaw": "WAW",
    "amsterdam schiphol": "AMS",
    "munich": "MUC",
    "frankfurt": "FRA",
    "hamburg": "HAM",
    "nice": "NCE",
    "lyon": "LYS",
    "porto": "OPO",
    "seville": "SVQ",
    "valencia": "VLC",
    "edinburgh": "EDI",
    "manchester": "MAN",
    "birmingham": "BHX",
    "glasgow": "GLA",
    "dublin ireland": "DUB",

    # ── Africa ────────────────────────────────────────────────
    "lagos": "LOS",
    "lagos nigeria": "LOS",
    "abuja": "ABV",
    "accra": "ACC",
    "nairobi": "NBO",
    "cairo": "CAI",
    "casablanca": "CMN",
    "marrakech": "RAK",
    "marrakesh": "RAK",
    "dakar": "DKR",
    "addis ababa": "ADD",
    "cape town": "CPT",
    "johannesburg": "JNB",
    "dar es salaam": "DAR",
    "kigali": "KGL",
    "kampala": "EBB",
    "entebbe": "EBB",
    "tunis": "TUN",
    "algiers": "ALG",
    "luanda": "LAD",
    "maputo": "MPM",
    "harare": "HRE",
    "lusaka": "LUN",
    "freetown": "FNA",
    "conakry": "CKY",
    "abidjan": "ABJ",
    "douala": "DLA",
    "yaoundé": "NSI",
    "yaounde": "NSI",
    "bamako": "BKO",
    "ouagadougou": "OUA",
    "libreville": "LBV",
    "malabo": "SSG",
    "bangui": "BGF",
    "brazzaville": "BZV",
    "kinshasa": "FIH",
    "antananarivo": "TNR",
    "port louis": "MRU",
    "mauritius": "MRU",
    "seychelles": "SEZ",
    "mahe": "SEZ",

    # ── Middle East ───────────────────────────────────────────
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "doha": "DOH",
    "riyadh": "RUH",
    "jeddah": "JED",
    "kuwait city": "KWI",
    "kuwait": "KWI",
    "muscat": "MCT",
    "beirut": "BEY",
    "amman": "AMM",
    "tel aviv": "TLV",

    # ── Asia ─────────────────────────────────────────────────
    "tokyo": "NRT",
    "tokyo narita": "NRT",
    "tokyo haneda": "HND",
    "singapore": "SIN",
    "hong kong": "HKG",
    "bangkok": "BKK",
    "kuala lumpur": "KUL",
    "seoul": "ICN",
    "incheon": "ICN",
    "beijing": "PEK",
    "shanghai": "PVG",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "jakarta": "CGK",
    "manila": "MNL",
    "taipei": "TPE",
    "osaka": "KIX",
    "bali": "DPS",
    "denpasar": "DPS",
    "colombo": "CMB",
    "kathmandu": "KTM",

    # ── North America ─────────────────────────────────────────
    "new york": "JFK",
    "new york city": "JFK",
    "nyc": "JFK",
    "los angeles": "LAX",
    "la": "LAX",
    "chicago": "ORD",
    "miami": "MIA",
    "toronto": "YYZ",
    "montreal": "YUL",
    "vancouver": "YVR",
    "cancun": "CUN",
    "mexico city": "MEX",
    "san francisco": "SFO",
    "seattle": "SEA",
    "atlanta": "ATL",
    "boston": "BOS",
    "washington": "IAD",
    "washington dc": "IAD",
    "dallas": "DFW",
    "houston": "IAH",
    "las vegas": "LAS",

    # ── South America ─────────────────────────────────────────
    "sao paulo": "GRU",
    "são paulo": "GRU",
    "rio de janeiro": "GIG",
    "rio": "GIG",
    "buenos aires": "EZE",
    "bogota": "BOG",
    "bogotá": "BOG",
    "lima": "LIM",
    "santiago": "SCL",
    "medellin": "MDE",
    "medellín": "MDE",

    # ── Oceania ───────────────────────────────────────────────
    "sydney": "SYD",
    "melbourne": "MEL",
    "brisbane": "BNE",
    "auckland": "AKL",

    # ── Aliases / shorthand ───────────────────────────────────
    "lon": "LHR",
    "dub": "DUB",
    "ist": "IST",
    "cdg": "CDG",
    "jfk": "JFK",
    "lhr": "LHR",
    "ams": "AMS",
    "fco": "FCO",
    "los": "LOS",
}


# Month name → number
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def resolve_iata(city_name: str) -> str | None:
    """Resolves a city name to IATA code. Returns None if unknown."""
    print(f"Resolving IATA for city: '{city_name}'")  # Debug print
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
 
# Region kgmids for arrival_area_id
REGIONS = {
    "europe": "/m/02j9z",
    "asia": "/m/0j0k",
    "north america": "/m/05rgl",
    "africa": "/m/0dg3n1",
    "caribbean": "/m/01p6xx",
    "middle east": "/m/03838",
    "south america": "/m/05r7t",
    "oceania": "/m/057ln",
}

COUNTRY_TO_CONTINENT = {
    # =========================
    # Africa
    # =========================
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Kenya": "Africa",
    "Egypt": "Africa",
    "Morocco": "Africa",
    "Ghana": "Africa",
    "Tanzania": "Africa",
    "Ethiopia": "Africa",
    "Tunisia": "Africa",
    "Algeria": "Africa",
    "Uganda": "Africa",
    "Rwanda": "Africa",

    # =========================
    # Europe
    # =========================
    "United Kingdom": "Europe",
    "Ireland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Portugal": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Austria": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Poland": "Europe",
    "Greece": "Europe",
    "Czech Republic": "Europe",
    "Hungary": "Europe",

    # =========================
    # Asia
    # =========================
    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "Singapore": "Asia",
    "Malaysia": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "United Arab Emirates": "Asia",
    "Saudi Arabia": "Asia",
    "Qatar": "Asia",
    "Turkey": "Asia",  # sometimes classified as Europe too depending on system

    # =========================
    # North America
    # =========================
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Costa Rica": "North America",
    "Cuba": "North America",
    "Jamaica": "North America",
    "Dominican Republic": "North America",

    # =========================
    # South America
    # =========================
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Peru": "South America",
    "Colombia": "South America",
    "Ecuador": "South America",
    "Bolivia": "South America",
    "Uruguay": "South America",

    # =========================
    # Oceania
    # =========================
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Fiji": "Oceania",
    "Papua New Guinea": "Oceania",

    # =========================
    # Antarctica
    # =========================
    "Antarctica": "Antarctica"
}


IATA_TO_COUNTRY: dict[str, str] = {
    # Europe
    "DUB": "Ireland",
    "LHR": "United Kingdom",
    "AMS": "Netherlands",
    "LIS": "Portugal",
    "IST": "Turkey",
    "CDG": "France",
    "FCO": "Italy",
    "MXP": "Italy",
    "BER": "Germany",
    "MAD": "Spain",
    "BCN": "Spain",
    "VIE": "Austria",
    "ATH": "Greece",
    "CPH": "Denmark",
    "ARN": "Sweden",
    # Africa
    "LOS": "Nigeria",
    "ABV": "Nigeria",
    "ACC": "Ghana",
    "NBO": "Kenya",
    "CAI": "Egypt",
    "CMN": "Morocco",
    "RAK": "Morocco",
    "DKR": "Senegal",
    "ADD": "Ethiopia",
    "CPT": "South Africa",
    "JNB": "South Africa",
    "DAR": "Tanzania",
    "KGL": "Rwanda",
    "EBB": "Uganda",
    "TUN": "Tunisia",
    "ALG": "Algeria",
    # North America
    "JFK": "United States",
    "YYZ": "Canada",
    "YUL": "Canada",
    "YVR": "Canada",
    # Asia / Middle East
    "DXB": "United Arab Emirates",
    "SIN": "Singapore",
    "NRT": "Japan",
    # Oceania
    "SYD": "Australia",
}
