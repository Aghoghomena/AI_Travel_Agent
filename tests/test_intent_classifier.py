import pytest
from unittest.mock import patch, MagicMock
from src.agents.intent_classifier import intent_classifier_agent
from src.agents.intent_classifier import (
    build_intent_classifier_graph,
    run_intent_classifier,
    classify_node,
    validate_node,
    retry_node,
    update_state_node,
    resolve_iata,
    validate_classification
)
from src.state import IntentStatus, QueryState, Traveller


# ═══════════════════════════════════════════════════════════════
# TOOL TESTS — resolve_iata
# ═══════════════════════════════════════════════════════════════

class TestResolveIata:

    def test_known_city_returns_iata(self):
        assert resolve_iata("Dublin") == "DUB"

    def test_case_insensitive(self):
        assert resolve_iata("LAGOS") == "LOS"
        assert resolve_iata("toronto") == "YYZ"
        assert resolve_iata("Toronto") == "YYZ"

    def test_unknown_city_returns_none(self):
        assert resolve_iata("Timbuktu") is None
        assert resolve_iata("") is None

    def test_all_seeded_cities_resolve(self):
        cities = {
            "Dublin": "DUB",
            "Lagos": "LOS",
            "Toronto": "YYZ",
            "Istanbul": "IST",
            "Lisbon": "LIS",
            "Dubai": "DXB",
            "Amsterdam": "AMS",
            "London": "LHR",
            "Tokyo": "NRT",
            "Sydney": "SYD",
            "Accra": "ACC",
            "Nairobi": "NBO",
            "Singapore": "SIN",
        }
        for city, expected_iata in cities.items():
            assert resolve_iata(city) == expected_iata, \
                f"Failed for {city}"
            

# ═══════════════════════════════════════════════════════════════
# TOOL TESTS — validate_classification
# ═══════════════════════════════════════════════════════════════

class TestValidateClassification:
    def test_valid_out_of_scope(self):
        result = {
            "status": "OUT_OF_SCOPE",
            "reason": "Not travel related"
        }
        is_valid, error = validate_classification(result)
        assert is_valid is True
        assert error == ""

    def test_out_of_scope_missing_reason(self):
        result = {"status": "OUT_OF_SCOPE"}
        is_valid, error = validate_classification(result)
        assert is_valid is False
        assert "reason" in error

    def test_valid_needs_info(self):
        result = {
            "status": "NEEDS_INFO",
            "missing_fields": ["travel_month"],
            "detected_fields": {
                "origins": ["Dublin"],
                "travel_month": None,
                "duration_nights": None,
                "region_preferences": None
            }
        }
        is_valid, error = validate_classification(result)
        assert is_valid is True

    def test_ready_missing_origins(self):
        result = {
            "status": "READY",
            "detected_fields": {
                "origins": [],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": None
            }
        }
        is_valid, error = validate_classification(result)
        assert is_valid is False
        assert "origins" in error

    def test_ready_missing_duration(self):
        result = {
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin"],
                "travel_month": "June",
                "duration_nights": None,
                "region_preferences": None
            }
        }
        is_valid, error = validate_classification(result)
        assert is_valid is False
        assert "duration" in error

    def test_invalid_status(self):
        result = {"status": "MAYBE"}
        is_valid, error = validate_classification(result)
        assert is_valid is False

    def test_not_a_dict(self):
        is_valid, error = validate_classification("not a dict")
        assert is_valid is False

# ═══════════════════════════════════════════════════════════════
# NODE TESTS — individual nodes with mocked LLM
# ═══════════════════════════════════════════════════════════════

class TestClassifyNode:

    def _make_state(self, message: str) -> dict:
        return {
            "user_message": message,
            "raw_llm_output": None,
            "classification": None,
            "retry_count": 0,
            "error": None,
            "query_state": QueryState(),
            "intent_status": None,
            "out_of_scope_reason": None,
            "agent_response": None,
        }
    
    @patch("src.agents.intent_classifier.llm")
    def test_classify_node_writes_raw_output(self, mock_client):
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"status": "OUT_OF_SCOPE", "reason": "test"}')
        ]
        mock_client.invoke.return_value = mock_response

        state = self._make_state("Write me a CV")
        result = classify_node(state)

        assert result["raw_llm_output"] is not None
        assert "OUT_OF_SCOPE" in result["raw_llm_output"]

    @patch("src.agents.intent_classifier.llm")
    def test_classify_node_strips_markdown_fences(self, mock_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='```json\n{"status": "OUT_OF_SCOPE", '
                 '"reason": "test"}\n```'
        )]
        mock_client.invoke.return_value = mock_response

        state = self._make_state("anything")
        result = classify_node(state)

        assert not result["raw_llm_output"].startswith("```")

class TestValidateNode:

    def _base_state(self) -> dict:
        return {
            "user_message": "test",
            "raw_llm_output": None,
            "classification": None,
            "retry_count": 0,
            "error": None,
            "query_state": QueryState(),
            "intent_status": None,
            "out_of_scope_reason": None,
            "agent_response": None,
        }
    
    def test_valid_json_sets_classification(self):
        state = {
            **self._base_state(),
            "raw_llm_output": '{"status": "OUT_OF_SCOPE", '
                              '"reason": "not travel"}'
        }
        result = validate_node(state)
        assert result["classification"] is not None
        assert result["error"] is None

    def test_invalid_json_sets_error(self):
        state = {
            **self._base_state(),
            "raw_llm_output": "not valid json at all"
        }
        result = validate_node(state)
        assert result["classification"] is None
        assert result["error"] is not None

    def test_wrong_structure_sets_error(self):
        state = {
            **self._base_state(),
            "raw_llm_output": '{"status": "WRONG_VALUE"}'
        }
        result = validate_node(state)
        assert result["error"] is not None

class TestRetryNode:

    def _base_state(self, retry_count: int = 0) -> dict:
        return {
            "user_message": "test",
            "raw_llm_output": None,
            "classification": None,
            "retry_count": retry_count,
            "error": "some error",
            "query_state": QueryState(),
            "intent_status": None,
            "out_of_scope_reason": None,
            "agent_response": None,
        }

    def test_increments_retry_count(self):
        state = self._base_state(retry_count=0)
        result = retry_node(state)
        assert result["retry_count"] == 1

    def test_max_retries_sets_fallback_classification(self):
        state = self._base_state(retry_count=1)
        result = retry_node(state)
        assert result["retry_count"] == 2
        assert result["classification"] is not None
        assert result["classification"]["status"] == "NEEDS_INFO"
        assert result["error"] is None

    def test_fallback_has_all_missing_fields(self):
        state = self._base_state(retry_count=1)
        result = retry_node(state)
        missing = result["classification"]["missing_fields"]
        assert "travellers" in missing
        assert "travel_month" in missing
        assert "duration_nights" in missing

class TestUpdateStateNode:

    def _base_state(self, classification: dict) -> dict:
        return {
            "user_message": "test",
            "raw_llm_output": None,
            "classification": classification,
            "retry_count": 0,
            "error": None,
            "query_state": QueryState(),
            "intent_status": None,
            "out_of_scope_reason": None,
            "agent_response": None,
        }

    def test_out_of_scope_sets_status_and_response(self):
        state = self._base_state({
            "status": "OUT_OF_SCOPE",
            "reason": "Not travel related",
            "detected_fields": {}
        })
        result = update_state_node(state)
        assert result["intent_status"] == IntentStatus.OUT_OF_SCOPE
        assert result["agent_response"] is not None
        assert result["out_of_scope_reason"] == "Not travel related"

    def test_ready_sets_correct_status(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin", "Lagos"],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        assert result["intent_status"] == IntentStatus.READY

    def test_origins_added_to_query_state(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin", "Lagos"],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        cities = [
            t.origin_city
            for t in result["query_state"].travellers
        ]
        assert "Dublin" in cities
        assert "Lagos" in cities

    def test_iata_codes_resolved_on_update(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin", "Toronto"],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        iatas = [
            t.origin_iata
            for t in result["query_state"].travellers
        ]
        assert "DUB" in iatas
        assert "YYZ" in iatas

    def test_unknown_city_iata_is_none(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Timbuktu"],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        iatas = [
            t.origin_iata
            for t in result["query_state"].travellers
        ]
        assert None in iatas

    def test_does_not_duplicate_existing_origins(self):
        existing_query_state = QueryState(
            travellers=[
                Traveller(
                    name="Traveller 1",
                    origin_city="Dublin",
                    origin_iata="DUB"
                )
            ]
        )
        state = {
            "user_message": "test",
            "raw_llm_output": None,
            "classification": {
                "status": "NEEDS_INFO",
                "missing_fields": ["travel_month"],
                "detected_fields": {
                    "origins": ["Dublin", "Lagos"],
                    "travel_month": None,
                    "duration_nights": None,
                    "region_preferences": None
                }
            },
            "retry_count": 0,
            "error": None,
            "query_state": existing_query_state,
            "intent_status": None,
            "out_of_scope_reason": None,
            "agent_response": None,
        }
        result = update_state_node(state)
        cities = [
            t.origin_city.lower()
            for t in result["query_state"].travellers
        ]
        assert cities.count("dublin") == 1

    def test_travel_month_merged_into_query_state(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin"],
                "travel_month": "July",
                "duration_nights": 5,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        assert result["query_state"].travel_month == "July"

    def test_duration_merged_into_query_state(self):
        state = self._base_state({
            "status": "READY",
            "detected_fields": {
                "origins": ["Dublin"],
                "travel_month": "June",
                "duration_nights": 4,
                "region_preferences": None
            }
        })
        result = update_state_node(state)
        assert result["query_state"].duration_nights == 4

    def test_region_preference_merged(self):
        state = self._base_state({
            "status": "NEEDS_INFO",
            "missing_fields": ["origins"],
            "detected_fields": {
                "origins": [],
                "travel_month": "June",
                "duration_nights": 3,
                "region_preferences": "Europe"
            }
        })
        result = update_state_node(state)
        assert result["query_state"].region_preferences == "Europe"


# ═══════════════════════════════════════════════════════════════
# GRAPH TESTS — full graph with mocked LLM
# ═══════════════════════════════════════════════════════════════
class TestIntentClassifierGraph:
    def _invoke(self, message: str, llm_response: str) -> dict:
        """Helper — runs graph with mocked LLM response."""
        with patch("src.agents.intent_classifier.llm") as mock:
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text=llm_response)]
            mock.invoke.return_value = mock_resp

            return intent_classifier_agent.invoke({
                "user_message": message,
                "raw_llm_output": None,
                "classification": None,
                "retry_count": 0,
                "error": None,
                "query_state": QueryState(),
                "intent_status": None,
                "out_of_scope_reason": None,
                "agent_response": None,
            })
        
    def test_graph_reaches_end_on_valid_out_of_scope(self):
        result = self._invoke(
            "Write me a CV",
            '{"status": "OUT_OF_SCOPE", "reason": "Not travel"}'
        )
        assert result["intent_status"] == IntentStatus.OUT_OF_SCOPE

    def test_graph_reaches_end_on_valid_ready(self):
        result = self._invoke(
            "Dublin, Lagos, June, 3 nights",
            '{"status": "READY", "detected_fields": {'
            '"origins": ["Dublin", "Lagos"], '
            '"travel_month": "June", '
            '"outbound_date": null, '
            '"duration_nights": 3, '
            '"region_preference": null}}'
        )
        assert result["intent_status"] == IntentStatus.READY

    def test_graph_retries_on_bad_json_then_fallback(self):
        """Graph should retry twice then fall back to NEEDS_INFO."""
        with patch("src.agents.intent_classifier.llm") as mock:
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="not valid json")]
            mock.invoke.return_value = mock_resp

            result = intent_classifier_agent.invoke({
                "user_message": "something",
                "raw_llm_output": None,
                "classification": None,
                "retry_count": 0,
                "error": None,
                "query_state": QueryState(),
                "intent_status": None,
                "out_of_scope_reason": None,
                "agent_response": None,
            })

        assert result["intent_status"] == IntentStatus.NEEDS_INFO
        assert result["retry_count"] == 2

    def test_graph_populates_query_state_on_ready(self):
        result = self._invoke(
            "Dublin, Lagos, Toronto, June, 3 nights",
            '{"status": "READY", "detected_fields": {'
            '"origins": ["Dublin", "Lagos", "Toronto"], '
            '"travel_month": "June", '
            '"outbound_date": null, '
            '"duration_nights": 3, '
            '"region_preferences": null}}'
        )
        travellers = result["query_state"].travellers
        assert len(travellers) == 3
        cities = [t.origin_city for t in travellers]
        assert "Dublin" in cities
        assert "Lagos" in cities
        assert "Toronto" in cities

    def test_out_of_scope_sets_agent_response(self):
        result = self._invoke(
            "What is the weather in Paris?",
            '{"status": "OUT_OF_SCOPE", '
            '"reason": "Weather queries are out of scope"}'
        )
        assert result["agent_response"] is not None
        assert len(result["agent_response"]) > 0

# ═══════════════════════════════════════════════════════════════
# INTEGRATION TESTS — run_intent_classifier node function
# Tests the LangGraph node that plugs into the main graph
# ═══════════════════════════════════════════════════════════════

class TestRunIntentClassifierNode:

    def _make_main_state(self, message: str) -> dict:
        return {
            "user_message": message,
            "query_state": QueryState(),
            "intent_status": None,
            "agent_response": None,
            "out_of_scope_reason": None,
            "conversation_history": [],
        }

    @patch("src.agents.intent_classifier.llm")
    def test_node_returns_intent_status(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(
            text='{"status": "OUT_OF_SCOPE", "reason": "test"}'
        )]
        mock_client.invoke.return_value = mock_resp

        state = self._make_main_state("Write a poem")
        result = run_intent_classifier(state)

        assert "intent_status" in result
        assert result["intent_status"] == IntentStatus.OUT_OF_SCOPE

    @patch("src.agents.intent_classifier.llm")
    def test_node_returns_updated_query_state(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=(
            '{"status": "READY", "detected_fields": {'
            '"origins": ["Dublin"], '
            '"travel_month": "June", '
            '"outbound_date": null, '
            '"duration_nights": 3, '
            '"region_preferences": null}}'
        ))]
        mock_client.invoke.return_value = mock_resp

        state = self._make_main_state("Dublin, June, 3 nights")
        result = run_intent_classifier(state)

        assert "query_state" in result
        assert len(result["query_state"].travellers) == 1

    @patch("src.agents.intent_classifier.llm")
    def test_node_does_not_return_internal_state(self, mock_client):
        """
        Node should only return keys relevant to main graph.
        Internal classifier state should not leak out.
        """
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(
            text='{"status": "OUT_OF_SCOPE", "reason": "test"}'
        )]
        mock_client.invoke.return_value = mock_resp

        state = self._make_main_state("anything")
        result = run_intent_classifier(state)

        assert "raw_llm_output" not in result
        assert "retry_count" not in result
        assert "error" not in result