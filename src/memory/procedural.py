from datetime import datetime

from requests import get
from src.memory.db import get_db_connection

class ProceduralMemory:

    def get_top_destinations_for_origin_pattern(self, origins: list[str], top_n: int = 5) -> list[dict]:
        """Returns destinations that have historically won
        for this origin pattern, sorted by win rate.
        Used by orchestrator to prioritise candidate destinations."""
        origin_pattern = "-".join(sorted(o.lower() for o in origins))
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT destination, win_count, total_searches, ROUND(CAST(win_count AS REAL) /total_searches * 100, 1) AS win_rate, avg_savings_usd, last_updated
            FROM procedural_patterns
            WHERE origin_pattern = ?
            ORDER BY win_rate DESC, win_count DESC
            LIMIT ?
        """, (origin_pattern, top_n))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def record_winner(self, origins: list[str], winning_destination: str, all_destinations: list[str], savings_vs_second_usd: float) -> None:
        """Updates procedural memory after a completed search.
        Increments win count for winner, total_searches for all patterns with same origins, and updates average savings."""
        origin_pattern = "-".join(sorted(o.lower() for o in origins))
        now = datetime.utcnow().isoformat()
        conn = get_db_connection()

        for dest in all_destinations:
            is_winner = dest.lower() == winning_destination.lower()
            conn.execute("""
                INSERT INTO procedural_patterns
                    (origins_pattern, destination, win_count,
                     total_searches, avg_saving_usd, last_updated)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(origins_pattern, destination)
                DO UPDATE SET
                    win_count = win_count + ?,
                    total_searches = total_searches + 1,
                    avg_saving_usd = (avg_saving_usd +
                                     excluded.avg_saving_usd) / 2,
                    last_updated = excluded.last_updated
            """, (
                origin_pattern,
                dest.lower(),
                1 if is_winner else 0,
                savings_vs_second_usd if is_winner else 0.0,
                now,
                1 if is_winner else 0
            ))

        conn.commit()
        conn.close()



    
    def get_all_patterns(self) -> list[dict]:
        """Returns all learned patterns — useful for debugging."""
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT * FROM procedural_patterns
            ORDER BY win_count DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]