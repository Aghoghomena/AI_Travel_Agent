"""
Run once to populate semantic memory with destination knowledge.
Usage: uv run python -m src.memory.seed
"""

import json
from pathlib import Path
from src.memory.db import initialize_db
from src.memory.semantic import SemanticMemory

def seed_destinations()-> None:
    """Seeds the semantic memory with destination knowledge already existing on a JSON file."""
    seed_path = Path("data/semantic_seed/destinations.json")
    if not seed_path.exists():
        print(f"Seed file not found at {seed_path}")
        return

    with open(seed_path) as f:
        destinations = json.load(f)

    mem = SemanticMemory()
    for dest in destinations:
        mem.add_destination(dest)
        print(f"  Seeded: {dest['city']} ({dest['iata']})")

    print(f"\nSeeded {len(destinations)} destinations into ChromaDB")

if __name__ == "__main__":
    print("Initializing database...")
    initialize_db()
    print("Seeding semantic memory with destination knowledge from json file...")
    seed_destinations()
    print("Done.")