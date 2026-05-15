import json
from datetime import datetime, timedelta
from src.memory.db import get_db_connection
from src.state import FlightResult, AccommodationResult
import os

CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))
VOLATILE_TTL_HOURS = int(os.getenv("VOLATILE_CURRENCY_TTL_HOURS", 6))
VOLATILE_CURRENCIES = {"NGN", "TRY", "ARS", "ZWL"}

def sort_origins_key(origins: list[str]) -> str:
    """Sorts origins so Dublin+Lagos == Lagos+Dublin. Also lowercases to avoid case-sensitive mismatches. to avoid cache misses due to different casing and ordering of origins."""
    return "-".join(sorted(o.lower() for o in origins))

def ttl_hours(currencies: list[str]) -> int:
    """Shorter Time To Live if any volatile currency is involved."""
    if any(c in VOLATILE_CURRENCIES for c in currencies):
        return VOLATILE_TTL_HOURS
    return CACHE_TTL_HOURS

class EpisodicMemory:

    # ── Flight leg cache ──────────────────────────────────────

    def get_flight_leg(self, origin_iata: str, destination_iata: str, outbound_date: str) -> FlightResult | None:
        """Returns cached flight leg result if available and not expired."""
        conn = get_db_connection()
        cursor = conn.cursor()
        row = cursor.execute("""
            SELECT * FROM flights
            WHERE origin_iata = ? AND destination_iata = ? AND outbound_date = ? AND expires_at > ?
        """, (origin_iata, destination_iata, outbound_date, datetime.utcnow())).fetchone()
        conn.close()
        if row:
                return FlightResult(
                    origin_iata=origin_iata,
                    destination_iata=destination_iata,
                    price_local=row["price_local"],
                    currency=row["currency"],
                    price_usd=row["price_usd"],
                    fetched_at=row["fetched_at"],
                    from_cache=True
                )
        return None

    def set_flight_leg(self, result: FlightResult):
        """Inserts or updates a flight leg result in the cache."""
        conn = get_db_connection()
        cursor = conn.cursor()
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours([result.currency]))
        cursor.execute("""
            INSERT INTO flights (origin_iata, destination_iata, outbound_date, price_local, currency, price_usd, fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(origin_iata, destination_iata, outbound_date) DO UPDATE SET
                price_local=excluded.price_local,
                currency=excluded.currency,
                price_usd=excluded.price_usd,
                fetched_at=excluded.fetched_at,
                expires_at=excluded.expires_at
        """, (
            result.origin_iata,
            result.destination_iata,
            result.outbound_date,
            result.price_local,
            result.currency,
            result.price_usd,
            result.fetched_at,
            expires_at.isoformat()
        ))
        conn.commit()
        conn.close()



 # ── Full group search cache ───────────────────────────────
    def get_full_group_search(self, origins: list[str], destinations: list[str], outbound_date: str, duration_nights: int) -> dict | None:
        """Returns cached full search result if available and not expired."""
        conn = get_db_connection()
        cursor = conn.cursor()
        origins_key = sort_origins_key(origins)
        row = cursor.execute("""
            SELECT * FROM episodic_searches
            WHERE origins_key = ? AND destinations = ? AND outbound_date = ? AND duration_nights = ? AND expires_at > ?
        """, (origins_key, json.dumps(destinations), outbound_date, duration_nights, datetime.utcnow())).fetchone()
        conn.close()
        if row:
            return {
                "origins": row["origins_key"],
                "destinations": json.loads(row["destinations"]),
                "outbound_date": row["outbound_date"],
                "duration_nights": row["duration_nights"],
                "flights": json.loads(row["flights_json"]),
                "accommodation": json.loads(row["accommodation_json"]),
                "total_usd": row["total_usd"],
                "exchange_rates": json.loads(row["exchange_rates_json"]),
                "fetched_at": row["fetched_at"],
                "from_cache": True
            }
        return None

    def set_group_search(self,origins: list[str],destination: str,outbound_date: str,duration_nights: int,flights: list[FlightResult],accommodation: AccommodationResult,total_usd: float,exchange_rates: dict) -> None:
        """Stores a complete group search result in memory."""
        key = sort_origins_key(origins)
        now = datetime.utcnow()
        currencies = [f.currency for f in flights]
        ttl = ttl_hours(currencies)
        expires = (now + timedelta(hours=ttl)).isoformat()

        flights_json = json.dumps([
            {
                "origin_iata": f.origin_iata,
                "destination_iata": f.destination_iata,
                "price_local": f.price_local,
                "currency": f.currency,
                "price_usd": f.price_usd,
            }
            for f in flights
        ])

        acc_json = json.dumps({
            "destination_city": accommodation.destination_city,
            "price_per_night_local": accommodation.price_per_night_local,
            "currency": accommodation.currency,
            "nights": accommodation.nights,
            "total_usd": accommodation.total_usd,
        })

        conn = get_db_connection()
        conn.execute("""
            INSERT INTO episodic_searches
                (origins_key, destinations, outbound_date,
                 duration_nights, flights_json, accommodation_json,
                 total_usd, exchange_rates_json,
                 fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key, json.dumps([destination.lower()]), outbound_date,
            duration_nights, flights_json, acc_json,
            total_usd, json.dumps(exchange_rates),
            now.isoformat(), expires
        ))
        conn.commit()
        conn.close()
    

    def get_past_searches(self, origins: list[str]) -> list[dict]:
        """Returns the 5 most recent past searches for an origin combination."""
        key = sort_origins_key(origins)
        conn = get_db_connection()
        rows = conn.execute(""" SELECT destinations, outbound_date, total_usd, fetched_at
            FROM episodic_searches WHERE origins_key = ? ORDER BY fetched_at DESC LIMIT 5
        """, (key,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
