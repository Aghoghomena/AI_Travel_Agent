import os
import json
from pathlib import Path
from certifi import where
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma")
AI_KEY = os.getenv("DEEPSEEK_API_KEY")


class SemanticMemory:

    def __init__(self):
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._ef = embedding_functions.DefaultEmbeddingFunction()

         # Two collections
        self.destinations = self.client.get_or_create_collection(
            name="destinations",
            embedding_function=self._ef
        )

        self.activities = self.client.get_or_create_collection(
            name="activities",
            embedding_function=self._ef
        )

        # ── Destination knowledge ─────────────────────────────────

    def add_destination(self, destination: str):
        """
        Adds or updates a destination in semantic memory.
        destination dict must have at minimum: city, iata, region
        """

        doc = json.dumps(destination)

        self.destinations.upsert(
            ids=[destination["iata"]],
            documents=[doc],
            metadatas=[{
                "city": destination["city"],
                "iata": destination["iata"],
                "region": destination.get("region", ""),
                "climate_tags": "".join(destination.get("climate_tags", []))
            }]
        )

    def query_destinations(self, query: str, region_filter: str | None = None, n_results: int = 5) -> list[dict]:
        """
        Semantic search over destinations.
        e.g. query='warm affordable city June'
        """

        where = None
        if region_filter and region_filter.lower() != "any":
            where = {"region": region_filter.lower()}

        results = self.destinations.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        destinations = []
        for doc in results["documents"][0]:
            try:
                destinations.append(json.loads(doc))
            except json.JSONDecodeError:
                continue

        return destinations
    
    def get_destination(self, iata: str) -> dict | None:
        """Retrieves a destination by its IATA code."""
        results = self.destinations.get(ids=[iata])
        if results["documents"]:
            try:
                return json.loads(results["documents"][0])
            except json.JSONDecodeError:
                return None
        return None
    
    # ── Activity knowledge ────────────────────────────────────

    def get_activities(self, destination_iata: str) -> dict | None:
        """Retrieves activities for a destination."""
        results = self.activities.get(where={"destination_iata": destination_iata})
        activities = []
        if results["documents"]:
            for doc in results["documents"]:
                try:
                    activities.append(json.loads(doc))
                except json.JSONDecodeError:
                    continue
        return activities           
    
    def add_activities(self, destination_iata: str, activities: list[dict]) -> None:
        """Adds or updates activities for a destination."""
        for i, activity in enumerate(activities):
            activity["destination_iata"] = destination_iata
            self.activities.upsert(
                ids=[f"{destination_iata}-activity-{i}"],
                documents=[json.dumps(activity)],
                metadatas=[{
                    "destination_iata": destination_iata,
                    "name": activity.get("name", ""),
                    "is_free": str(activity.get("is_free", False)),
                }]
            )
        