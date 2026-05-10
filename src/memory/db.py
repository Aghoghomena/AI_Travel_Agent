import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/travel_agent.db")

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    # Ensure the directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
    conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
    return conn

def initialize_db():
    """Initializes the database with the necessary tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Episodic: full group search results ───────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origins_key TEXT NOT NULL,
            destinations TEXT NOT NULL,
            outbound_date TEXT NOT NULL,
            return_date TEXT NOT NULL,
            duration_nights INTEGER NOT NULL,
            flights_json TEXT NOT NULL,
            accommodation_json TEXT NOT NULL,
            activities_json TEXT NOT NULL,
            total_usd REAL NOT NULL,
            exchange_rate REAL NOT NULL,
            fetched_at text NOT NULL,
            expires_at text NOT NULL
        )
    """)

    # ── Episodic: individual flight leg cache ─────────────────

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_iata TEXT NOT NULL,
            destination_iata TEXT NOT NULL,
            outbound_date TEXT NOT NULL,
            price_local REAL NOT NULL,
            currency TEXT NOT NULL,
            price_usd REAL NOT NULL,
            fetched_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            UNIQUE(origin_iata, destination_iata, outbound_date)
        )
    """)


    # ── Procedural: winning destination patterns ──────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS procedural_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_pattern TEXT NOT NULL,
            destination TEXT NOT NULL,
            win_count INTEGER DEFAULT 1,
            total_searches INTEGER DEFAULT 1,
            avg_savings_usd REAL DEFAULT 0.0,
            last_updated TEXT NOT NULL,
            UNIQUE(origin_pattern, destination)
        )
    """)

    # ── HITL audit log ────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hitl_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            message TEXT NOT NULL,
            user_response TEXT,
            correction_json TEXT,
            logged_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully at", DB_PATH)