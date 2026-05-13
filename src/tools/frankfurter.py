"""
Currency exchange rate tool.
 
Uses fawazahmed0/exchange-api — free, no key, 200+ currencies,
daily updated, covers all African currencies.
 
Primary:  https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json
Fallback: https://latest.currency-api.pages.dev/v1/currencies/usd.json
 
Endpoint returns USD as base with all target currencies in one call.
No conversion endpoint — fetch rate and multiply locally.
"""

import os
import httpx
from datetime import datetime
from data.iata import IATA_TO_CURRENCY

 
PRIMARY_URL  = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
FALLBACK_URL = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
TIMEOUT = 10.0

def get_currency_for_iata(iata: str) -> str:
    """Returns the local currency code for a given origin IATA."""
    return IATA_TO_CURRENCY.get(iata.upper(), "USD")

def fetch_rates(currencies: list[str]) -> dict[str, float]:
    """
    Fetches USD → target currency rates.
    Makes one API call and filters to requested currencies.
    Tries primary URL first, falls back to Cloudflare mirror.
    Returns dict of {currency_code: rate}.
    """
    rates: dict[str, float] = {"USD": 1.0}
    currencies_lower = {c.lower() for c in currencies if c != "USD"}
 
    for url in [PRIMARY_URL, FALLBACK_URL]:
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                # Response shape: {"date": "...", "usd": {"eur": 0.92, "ngn": 1580, ...}}
                all_rates = data.get("usd", {})
                for currency in currencies:
                    key = currency.lower()
                    if key in all_rates:
                        rates[currency] = float(all_rates[key])
                    elif currency != "USD":
                        print(f"[currency] Rate not found for {currency}")
                return rates
        except Exception as e:
            print(f"[currency] Error fetching from {url}: {e}")
 
    print("[currency] Both URLs failed — using USD=1.0 fallback for all")
    return rates

def convert_usd_to_local(amount_usd: float, currency: str, rates: dict[str, float]) -> float:
    """Converts a USD amount to local currency using the provided rates."""
    rate = rates.get(currency, 1.0)
    return round(amount_usd * rate, 2)
 