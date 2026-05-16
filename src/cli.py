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
import xml.etree.ElementTree as ET
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

def _step_label(step: ET.Element) -> str:
    tool = step.get("tool", "")
    if tool == "memory_read":
        return "Check past searches for cached results"
    if tool == "traveller_agent":
        name = step.get("traveller", "")
        origin = step.get("origin", "")
        dests = step.get("destinations", "").replace(",", ", ")
        return f"Find flights for {name} ({origin} → {dests})"
    if tool == "accommodation_agent":
        dests = step.get("destinations", "").replace(",", ", ")
        return f"Find hotels in {dests}"
    if tool == "currency_agent":
        return "Convert all prices to USD"
    if tool == "ranker_agent":
        return "Rank destinations by total group cost"
    if tool == "memory_write":
        return "Save results for next time"
    return tool


def _execution_groups(steps: dict) -> list[list[str]]:
    groups, completed, remaining = [], set(), set(steps)
    while remaining:
        ready = [
            sid for sid in remaining
            if all(d.strip() in completed for d in steps[sid].get("depends_on", "").split(",") if d.strip())
        ]
        if not ready:
            break
        groups.append(ready)
        completed.update(ready)
        remaining -= set(ready)
    return groups


def print_plan(plan_xml: str):
    try:
        root = ET.fromstring(plan_xml.strip())
    except ET.ParseError:
        return

    steps = {s.get("id"): s for s in root.findall("step")}
    groups = _execution_groups(steps)

    lines = []
    for group in groups:
        labels = [_step_label(steps[sid]) for sid in group if sid in steps]
        if len(labels) == 1:
            lines.append(f"  • {labels[0]}")
        else:
            lines.append(f"  • Simultaneously:")
            for label in labels:
                lines.append(f"      · {label}")

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Here's what I'm going to do[/bold cyan]",
        box=box.ROUNDED,
    ))


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
        # Activities
        dest_activities = activities.get(dest.iata, [])
        if dest_activities:
            console.print(f"    [bold]🎯  Things to Do[/bold]")
            by_category: dict[str, list] = {}
            for act in dest_activities:
                by_category.setdefault(act.category, []).append(act)
            for category, acts in by_category.items():
                console.print(f"       [cyan]{category}[/cyan]")
                for act in acts:
                    cost = "[green]Free[/green]" if act.is_free else f"[green]${act.estimated_cost_usd:.0f}[/green]"
                    rating = f"  [yellow]★ {act.rating}[/yellow]" if act.rating else ""
                    console.print(f"         · [white]{act.name}[/white]  {cost}{rating}")

 
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
        "query_state":          {},
        "elicitation_complete": False,
        "elicitation_question": None, 
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
                travel_agent_graph.invoke(Command(resume=user_input), config=config)
            else:
                state["user_message"] = user_input
                travel_agent_graph.invoke(state, config=config)

            # Always read real state from checkpointer
            result = travel_agent_graph.get_state(config)
            interrupt_data = result.interrupts  # ← interrupts live here on StateSnapshot
            interrupted = bool(interrupt_data)
            values = result.values  # ← actual state values

            # Show the plan the first time it appears, regardless of interrupt state
            plan = values.get("rewoo_plan")
            if plan and plan != state.get("rewoo_plan"):
                print_plan(plan)

            if interrupted:
                message = interrupt_data[0].value if interrupt_data else ""
                print_agent(message)
            else:
                intent = values.get("intent_status")

                if intent == "out_of_scope":
                    print_agent(values.get("agent_response", ""))
                elif values.get("elicitation_question") and not values.get("elicitation_complete"):
                    print_agent(values["elicitation_question"])
                elif values.get("activities_fetched"):
                    print_results(values.get("ranked_destinations", []), values.get("activities", {}))
                    break
                elif values.get("ranked_destinations"):
                    print_results(values.get("ranked_destinations", []), values.get("activities", {}))
                elif values.get("agent_response"):
                    print_agent(values["agent_response"])

                if values.get("errors"):
                    for e in values["errors"]:
                        if e:
                            console.print(f"[red]⚠[/red]  [dim]{e}[/dim]")

        except GraphInterrupt as gi:
            interrupted = True
            message = gi.args[0][0].value if gi.args and gi.args[0] else ""
            print_agent(message)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            interrupted = False
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
            

            
            
 
 
if __name__ == "__main__":
    app()