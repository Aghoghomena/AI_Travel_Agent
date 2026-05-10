import pytest
from src.agents.intent_classifier import IntentClassifierAgent
from src.state import IntentStatus

agent = IntentClassifierAgent()

# ── Out of scope ──────────────────────────────────────────────

def test_cv_writing_out_of_scope():
    result = agent.classify_intent("Write me a cover letter")
    assert result["status"] == "OUT_OF_SCOPE"


def test_weather_out_of_scope():
    result = agent.classify_intent("What is the weather in Paris?")
    assert result["status"] == "OUT_OF_SCOPE"


def test_booking_out_of_scope():
    result = agent.classify_intent("Book me a flight to London")
    assert result["status"] == "OUT_OF_SCOPE"


def test_restaurant_out_of_scope():
    result = agent.classify_intent("Recommend a good restaurant in Rome")
    assert result["status"] == "OUT_OF_SCOPE"


# ── Needs info ────────────────────────────────────────────────

def test_vague_travel_needs_info():
    result = agent.classify_intent("Me and friends want to meet somewhere")
    assert result["status"] == "NEEDS_INFO"


def test_partial_origins_needs_info():
    result = agent.classify_intent(
        "I'm in Dublin, friend in Lagos, where should we meet?"
    )
    assert result["status"] == "NEEDS_INFO"
    detected = result.get("detected_fields", {})
    assert "Dublin" in detected.get("origins", [])
    assert "Lagos" in detected.get("origins", [])


def test_region_no_origins_needs_info():
    result = agent.classify_intent(
        "Somewhere warm in Europe for 4 of us in July"
    )
    assert result["status"] == "NEEDS_INFO"
    assert "origins" in result.get("missing_fields", [])

# ── Ready ─────────────────────────────────────────────────────

def test_full_info_ready():
    result = agent.classify_intent(
        "Dublin, Lagos, Toronto, June, 3 nights"
    )
    assert result["status"] == "READY"


def test_natural_language_ready():
    result = agent.classify_intent(
        "Me in Dublin, friend in Lagos, travelling mid-June for 4 nights"
    )
    assert result["status"] == "READY"
    detected = result.get("detected_fields", {})
    assert detected.get("duration_nights") == 4


def test_multi_origin_ready():
    result = agent.classify_intent(
        "4 of us: Dublin, NYC, Tokyo, Sydney — July, one week"
    )
    assert result["status"] == "READY"
    detected = result.get("detected_fields", {})
    assert len(detected.get("origins", [])) >= 3


# ── Edge cases ────────────────────────────────────────────────

def test_empty_message():
    result = agent.classify_intent("")
    assert result["status"] in ["OUT_OF_SCOPE", "NEEDS_INFO"]


def test_single_origin_ready():
    result = agent.classify_intent(
        "Just me, flying from Dublin, June, 3 nights, Europe"
    )
    assert result["status"] == "READY"