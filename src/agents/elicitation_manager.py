import json
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from src.prompts.elicitation import build_elicitation_prompt
from src.state import IntentStatus, QueryState, Traveller
from src.utils.config import llm
from langgraph.types import interrupt, Command
from src.state import ElicitationState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

MAX_ELICITATION_TURNS = int(os.getenv("MAX_ELICITATION_TURNS", 5))


# ── Tools ─────────────────────────────────────────────────────
 
def validate_elicitation_output(parsed: dict) -> tuple[bool, str]:
    """
    Validates the LLM output is well-formed.
    Returns (is_valid, error_message)
    """
    if not isinstance(parsed, dict):
        return False, "Output is not a dict"
 
    if "question" not in parsed:
        return False, "Missing 'question' key"
 
    if "updated_fields" not in parsed:
        return False, "Missing 'updated_fields' key"
 
    if "all_required_present" not in parsed:
        return False, "Missing 'all_required_present' key"
 
    if not isinstance(parsed["updated_fields"], dict):
        return False, "'updated_fields' must be a dict"
 
    if not isinstance(parsed["all_required_present"], bool):
        return False, "'all_required_present' must be a bool"
 
    # If all required present, question should be empty
    if parsed["all_required_present"] and parsed["question"] != "":
        return False, "all_required_present is true but question is not empty"
 
    # Validate travellers list if present
    travellers = parsed["updated_fields"].get("travellers")
    if travellers is not None:
        if not isinstance(travellers, list):
            return False, "'travellers' must be a list"
        for t in travellers:
            if "origin_city" not in t:
                return False, "Each traveller must have 'origin_city'"
 
    return True, ""

def apply_updated_fields(query_state: QueryState, updated_fields: dict) -> QueryState:
    """
    Merges updated_fields from the LLM into the existing query_state.
    Returns the mutated query_state.
    """
    if "travellers" in updated_fields:
        existing_cities = {t.origin_city.lower() for t in query_state.travellers}
        new_travellers = []
        for i, t in enumerate(updated_fields["travellers"]):
            city = t.get("origin_city", "")
            if city.lower() not in existing_cities:
                new_travellers.append(
                    Traveller(
                        name=t.get("name", f"Traveller {i + 1}"),
                        origin_city=city,
                        origin_iata=t.get("origin_iata", ""),
                    )
                )
                existing_cities.add(city.lower())
        query_state.travellers.extend(new_travellers)
 
    if "travel_month" in updated_fields:
        query_state.travel_month = updated_fields["travel_month"]
 
    if "duration_nights" in updated_fields:
        val = updated_fields["duration_nights"]
        try:
            query_state.duration_nights = int(val)
        except (TypeError, ValueError):
            query_state.duration_nights = val
 
    if "region_preferences" in updated_fields:
        query_state.region_preferences = updated_fields["region_preferences"]
    
    if "outbound_date" in updated_fields:
        query_state.outbound_date = updated_fields["outbound_date"]

    if "return_date" in updated_fields:
        query_state.return_date = updated_fields["return_date"]
    
    if "destination" in updated_fields:
        # This is a bit hacky but allows us to pass back a selected destination from the LLM
        query_state.candidate_destinations = [
            d for d in query_state.candidate_destinations
            if d.city.lower() == updated_fields["destination"].lower()
        ]
    

    return query_state

def summarise_history(conversation_history: list[dict]) -> str:
    """
    Condenses conversation history into a single summary string.
    Only runs when there are 2+ turns in history.
    Returns a plain text summary, not a list of turns.
    """
    if not conversation_history:
        return ""
 
    lines = []
    for turn in conversation_history:
        role = "Agent" if turn["role"] == "assistant" else "User"
        lines.append(f"{role}: {turn['content']}")
 
    history_text = "\n".join(lines)
 
    prompt = (
        "Summarise this conversation in one or two sentences, "
        "focusing only on what travel information was collected "
        "(origins, month, nights, region) and any corrections made. "
        "Be concise.\n\n"
        f"{history_text}\n\nSummary:"
    )
 
    response = llm.invoke(prompt)
    return response.content.strip()


# ── Nodes ─────────────────────────────────────────────────────

def summarise_node(state: ElicitationState) -> ElicitationState:
    """
    Node 0: Summarises conversation history before the LLM call.
    Only runs when history has 2+ turns to keep context lean.
    Replaces the full history with a single summary entry.
    """
    history = state.get("conversation_history", [])
 
    if len(history) < 2:
        return state
 
    summary = summarise_history(history)
    condensed = [{"role": "assistant", "content": f"[Summary of conversation so far: {summary}]"}]
 
    return {**state, "conversation_history": condensed}
 
def elicit_node(state: ElicitationState) -> ElicitationState:
    """
    Node 1: Calls LLM to produce the next elicitation question
    and extract any fields from the user message.
    Writes raw output to state.
    """
    prompt = build_elicitation_prompt(
        user_message=state["user_message"],
        query_state=state.get("query_state", QueryState()),
        conversation_history=state.get("conversation_history", []),
    )
 
    response = llm.invoke(prompt)
    raw = response.content.strip()
 
    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
 
    return {**state, "raw_llm_output": raw}

def validate_node(state: ElicitationState) -> ElicitationState:
    """
    Node 2: Validates the raw LLM output.
    Parses JSON and checks structure.
    Writes parsed_output or error to state.
    """
    raw = state.get("raw_llm_output", "")
 
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            **state,
            "parsed_output": None,
            "error": f"JSON parse failed: {e}",
        }
 
    is_valid, error_msg = validate_elicitation_output(parsed)
 
    if not is_valid:
        return {
            **state,
            "parsed_output": None,
            "error": error_msg,
        }
 
    return {
        **state,
        "parsed_output": parsed,
        "error": None,
    }

def retry_node(state: ElicitationState) -> ElicitationState:
    """
    Node 3: Increments retry counter.
    LangGraph routes back to elicit_node if under limit.
    On max retries falls back to asking the first missing field.
    """
    retry_count = state.get("retry_count", 0) + 1
 
    if retry_count >= 2:
        query_state = state.get("query_state", QueryState())
        missing = query_state.missing_info()
        question = (
            f"Could you tell me {missing[0]}?"
            if missing
            else "Could you confirm your travel details?"
        )
        fallback = {
            "question": question,
            "updated_fields": {},
            "corrections": [],
            "all_required_present": False,
        }
        return {
            **state,
            "retry_count": retry_count,
            "parsed_output": fallback,
            "error": None,
        }
 
    return {**state, "retry_count": retry_count}

def update_state_node(state: ElicitationState) -> ElicitationState:
    """
    Node 4: Merges parsed LLM output into query_state.
    Sets elicitation_question and elicitation_complete on state.
    """
    parsed = state["parsed_output"]
    query_state = state.get("query_state", QueryState())
 
    updated_fields = parsed.get("updated_fields", {})
    query_state = apply_updated_fields(query_state, updated_fields)
 
    corrections = parsed.get("corrections", [])
    all_required_present = parsed.get("all_required_present", False)
    question = parsed.get("question", "")
    turn_count = state.get("turn_count", 0) + 1
    # Append this turn to history
    history = list(state.get("conversation_history", []))
    if state.get("user_message"):
        history.append({"role": "user", "content": state["user_message"]})
    if question:
        history.append({"role": "assistant", "content": question})
 
    return {
        **state,
        "query_state": query_state,
        "elicitation_question": question,
        "elicitation_complete": all_required_present,
        "corrections": corrections,
        "turn_count": turn_count,
        "conversation_history": history,
    }

def ask_user_node(state: ElicitationState) -> ElicitationState:
    """
    Node 5: Interrupts the graph to surface the question to the user
    and wait for their response. Resumes with user_message set.
    """
    user_response = interrupt(state["elicitation_question"])
    return {**state, "user_message": user_response, "retry_count": 0}


# ── Conditional edge functions ────────────────────────────────
 
def should_retry(state: ElicitationState) -> Literal["retry", "update"]:
    """Routes to retry if validation failed, update if valid."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    return "update"
 
 
def retry_or_elicit(state: ElicitationState) -> Literal["elicit", "update"]:
    """After retry node — go back to elicit or force update."""
    if state.get("parsed_output"):
        return "update"
    return "elicit"

def complete_or_continue(state: ElicitationState) -> Literal["ask_user", "end"]:
    """
    After update_state — end if complete or turn limit reached,
    otherwise pause and ask the user for more needed information.
    """
    if state.get("elicitation_complete"):
        return "end"
    if state.get("turn_count", 0) >= MAX_ELICITATION_TURNS:
        return "end"
    return "ask_user"

# ── Build the agent graph ─────────────────────────────────────
 
def build_elicitation_graph():
    graph = StateGraph(ElicitationState)
 
    # Add nodes
    graph.add_node("summarise", summarise_node)
    graph.add_node("elicit", elicit_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("update_state", update_state_node)
    graph.add_node("ask_user", ask_user_node)

    # Add edges
    graph.add_edge(START, "elicit")
    graph.add_edge("elicit", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "retry",
            "update": "update_state",
        },
    )
    graph.add_conditional_edges(
        "retry",
        retry_or_elicit,
        {
            "elicit": "elicit",
            "update": "update_state",
        },
    )

    graph.add_conditional_edges(
        "update_state",
        complete_or_continue,
        {"ask_user": "ask_user", "end": END},
    )

# After user answers, summarise and go again
    graph.add_edge("ask_user", "summarise")

    graph.add_edge("summarise", "elicit")
    
 
    return graph.compile(checkpointer=MemorySaver())

# Instantiate once at module level so it can be reused across calls
elicitation_agent = build_elicitation_graph()
try:
    elicitation_agent.get_graph().draw_mermaid_png(output_file_path="elicitation_agent.png")
    print("\nGraph saved as elicitation_agent.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")

# ── Entry point for main travel agent graph ───────────────────
 
def run_elicitation_manager(state: dict) -> dict:
    """
    Entry point for the main travel agent graph.
    Runs the full elicitation sub-graph with its own conversation loop.
    Uses a thread_id from state so the checkpointer can resume correctly.
    """
    config = {"configurable": {"thread_id": state.get("session_id", "default")}}
 
    result = elicitation_agent.invoke(
        {
            "user_message": state.get("user_message", ""),
            "raw_llm_output": None,
            "parsed_output": None,
            "retry_count": 0,
            "turn_count": 0,
            "error": None,
            "query_state": state.get("query_state", QueryState()),
            "elicitation_question": None,
            "elicitation_complete": False,
            "turn_limit_reached": False,
            "corrections": [],
            "conversation_history": state.get("conversation_history", []),
        },
        config=config,
    )

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        question = interrupts[0].value if interrupts else "Could you provide more details?"
        print(f"\nAgent: {question}")
        user_input = input("You: ").strip()
        result = elicitation_agent.invoke(Command(resume=user_input), config=config)

    turn_limit_reached = result.get("turn_count", 0) >= MAX_ELICITATION_TURNS and not result.get("elicitation_complete")
 
    return {
        "query_state": result["query_state"],
        "elicitation_question": result.get("elicitation_question"),
        "elicitation_complete": result.get("elicitation_complete", False),
        "turn_limit_reached": turn_limit_reached,
        "corrections": result.get("corrections", []),
        "conversation_history": result.get("conversation_history", []),
    }
 

 #test the agent manually

if __name__ == "__main__":

    # Single call — the agent owns the conversation loop internally
    result = run_elicitation_manager({
        "session_id": "test-session-1",
        "user_message": "Me and my want to travel together",
        "query_state": QueryState(),
        "conversation_history": [],
    })
    
    # When the agent finishes, print the final state
    print("\n=== Done ===")
    print(f"Complete:      {result['elicitation_complete']}")
    print(f"Turn limit:    {result['turn_limit_reached']}")
    print(f"Travellers:    {[t.origin_city for t in result['query_state'].travellers]}")
    print(f"Month:         {result['query_state'].travel_month}")
    print(f"Nights:        {result['query_state'].duration_nights}")
    print(f"Region:        {result['query_state'].region_preferences}")
