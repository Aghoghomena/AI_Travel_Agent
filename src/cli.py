"""
CLI Entry Point — Group Travel Agent.
 
Uses rich for formatted output and typer for CLI args.
Maintains session_id across conversation turns.
Feeds user input into the main LangGraph graph.
 
Usage:
  uv run python -m src.cli
  uv run python -m src.cli --session my-session
"""

import uuid
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
 
from src.graph import travel_agent_graph
from src.memory.seed import seed_destinations
from src.memory.db import initialize_db
from src.state import QueryState, IntentStatus
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
 
app = typer.Typer()
console = Console()
 


# ── Output formatting ─────────────────────────────────────────

def print_agent(message: str):
    console.print(f"\n[cyan]Agent:[/cyan] {message}\n")
 
def print_welcome():
    console.print(Panel(
        "[bold cyan]Group Travel Agent[/bold cyan]\n"
        "[dim]Find the cheapest city for your group to meet[/dim]\n\n"
        "[dim]Type 'quit' to exit, 'reset' to start over[/dim]",
        box=box.ROUNDED,
    ))

def print_out_of_scope(message: str):
    console.print(f"\n[yellow]ℹ[/yellow]  {message}\n")

def print_question(question: str):
    console.print(f"\n[cyan]Agent:[/cyan] {question}\n")

 
def print_results(ranked_destinations: list, activities: dict):
    if not ranked_destinations:
        console.print("\n[red]No results found.[/red]\n")
        return
 
    console.print("\n")
    for dest in ranked_destinations[:3]:
        rank_emoji = ["🏆", "🥈", "🥉"][dest.rank - 1] if dest.rank and dest.rank <= 3 else "•"
        total = dest.grand_total_usd or 0.0
 
        # Header
        console.print(
            f"{rank_emoji}  [bold]#{dest.rank} {dest.city} ({dest.iata})[/bold]"
            f"  —  [green]${total:,.0f} total group cost[/green]"
        )
 
        # Per-traveller breakdown
        if dest.per_traveller_breakdown:
            for b in dest.per_traveller_breakdown:
                console.print(
                    f"     [dim]{b.traveller_name} ({b.origin_iata} → {b.currency}):[/dim]"
                    f"  [white]{b.total_local:,.0f} {b.currency}[/white]"
                    f"  [dim](${b.total_usd:,.0f})[/dim]"
                )
 
        # Accommodation
        if dest.accommodation:
            a = dest.accommodation
            console.print(
                f"     [dim]Hotel:[/dim]  ${a.price_per_night_usd:.0f}/night"
                f" × {a.nights} nights = [white]${a.total_usd:,.0f}[/white]"
            )
 
        # Activities
        dest_activities = activities.get(dest.iata, [])
        if dest_activities:
            console.print(f"     [dim]Things to do:[/dim]")
            for act in dest_activities:
                cost = "Free" if act.is_free else f"${act.estimated_cost_usd:.0f}"
                dist = f"{act.distance_km:.1f} km" if act.distance_km else ""
                console.print(
                    f"       [dim]·[/dim] [white]{act.name:<30}[/white]"
                    f"  [green]{cost:<8}[/green]  [dim]{dist}[/dim]"
                )
 
        console.print()
 
 
def print_past_searches(summary: str):
    if summary:
        console.print(f"\n[dim]📋 {summary}[/dim]")
 
 
def print_error(errors: list[str]):
    for e in errors:
        if e:
            console.print(f"[red]⚠[/red]  [dim]{e}[/dim]")
 

# ── Main conversation loop ────────────────────────────────────
 
@app.command()
def main():
    """Group Travel Agent — find the cheapest city for your group to meet."""
 
    initialize_db()
    seed_destinations()
    print_welcome()
 
    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}
    console.print(f"[dim]Session: {session_id}[/dim]\n")
 
    # Initial state
    state = {
        "session_id":           session_id,
        "conversation_history": [],
        "user_message":         "",
        "agent_response":       None,
        "intent_status":        None,
        "query_state":          [],
        "elicitation_complete": False,
        "flight_results":       [],
        "accommodation_results":[],
        "destination_results":  [],
        "ranked_destinations":  [],
        "activity_destinations":[],
        "activities":           {},
        "episodic_cache_hit":   False,
        "memory_write_complete":False,
        "awaiting_hitl":        False,
        "errors":               [],
    }
 
    interrupted = False   # tracks whether graph is paused at interrupt()
 
    while True:
        try:
            user_input = console.input("[bold]You:[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
 
        if not user_input:
            continue
        if user_input.lower() == "quit":
            console.print("[dim]Goodbye.[/dim]")
            break
 
        try:
            if interrupted:
                # Graph is paused at interrupt() — resume with user's response
                result = travel_agent_graph.invoke(
                    Command(resume=user_input),
                    config=config,
                )
            else:
                # Normal invoke — new user message
                state["user_message"] = user_input
                result = travel_agent_graph.invoke(state, config=config)
 
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            interrupted = False
            continue
 
        # Always merge non-interrupt state first
        interrupt_data = result.get("__interrupt__")
        interrupted = bool(interrupt_data)
 
        # Merge result into state — never overwrite with None or empty values
        for k, v in result.items():
            if k == "__interrupt__":
                continue
            if v is None and state.get(k) is not None:
                continue
            if v == [] and state.get(k):
                continue
            if v == {} and state.get(k):
                continue
            state[k] = v
 
        if interrupted:
            # Graph paused at interrupt() — surface the message
            message = interrupt_data[0].value if interrupt_data else ""
            print_agent(message)
        else:
            # Graph ran to completion this turn
            intent = result.get("intent_status")
 
            if intent == IntentStatus.OUT_OF_SCOPE:
                print_agent(result.get("agent_response", ""))
 
            elif result.get("elicitation_question") and not result.get("elicitation_complete"):
                print_agent(result["elicitation_question"])
 
            elif result.get("ranked_destinations"):
                if result.get("past_searches_summary"):
                    console.print(f"\n[dim]{result['past_searches_summary']}[/dim]")
                print_results(
                    result.get("ranked_destinations", []),
                    result.get("activities", {}),
                )
 
            elif result.get("agent_response"):
                print_agent(result["agent_response"])
 
            if result.get("errors"):
                for e in result["errors"]:
                    if e:
                        console.print(f"[red]⚠[/red]  [dim]{e}[/dim]")
 
 
if __name__ == "__main__":
    app()