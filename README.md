# Group Travel Agent

An AI-powered travel planning assistant that finds the cheapest destination city for a group of people flying from **different origin cities**. Tell it who is travelling and from where, and it searches flights, hotels, and activities across candidate destinations to produce a ranked, cost-compared shortlist — with full per-traveller breakdowns in local currencies.

---

## How It Works

The agent runs as a multi-turn conversation in your terminal. It guides you through three stages:

### 1. Intent & Elicitation
The agent reads your initial message, extracts travel details (travellers, origins, dates, budget), and asks clarifying questions for anything missing. When it has enough information, it asks you to confirm before proceeding.

### 2. Planning & Execution
The agent generates a ReWOO execution plan — an ordered list of steps covering flight searches, hotel lookups, currency conversion, and ranking. You can review and approve the plan before any API calls are made. Steps that don't depend on each other run in parallel.

### 3. Results & Activities
The top three destinations are ranked by total group cost. You choose how many to drill into, and the agent fetches top-rated tourist activities for each.

At every major decision point (confirming details, approving the plan, selecting destinations) the agent pauses and waits for your input.

---

## Architecture

```
CLI (rich terminal UI)
  └─ LangGraph State Machine
       ├─ Intent Classifier     — parse user message, elicit missing fields
       ├─ Orchestrator          — generate ReWOO XML execution plan
       ├─ Traveller Agent × N   — search flights per origin (parallel)
       ├─ Accommodation Agent   — hotel costs per destination
       ├─ Currency Agent        — convert all prices to USD + local breakdown
       ├─ Ranker Agent          — rank destinations by total group cost
       ├─ Activities Agent      — fetch top attractions (Google Places)
       └─ Memory Agent          — cache reads/writes (episodic + procedural)

Memory System
  ├─ Episodic   (SQLite)   — recent flight & hotel search results (24h TTL)
  ├─ Procedural (SQLite)   — historical winning destinations per origin pattern
  └─ Semantic   (ChromaDB) — destination knowledge base + activity cache

External APIs
  ├─ SearchAPI → Amadeus   — live flight search
  ├─ Google Places API     — tourist activities
  └─ Frankfurter API       — currency exchange rates (no key required)
```

---

## Requirements

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd group-travel-agent
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Open `.env` and set the required values:

```env
# Required — LLM backend
DEEPSEEK_API_KEY=your_deepseek_key

# Required — Flight search (SearchAPI wrapper over Amadeus)
SEARCH_API_KEY=your_searchapi_key

# Required — Tourist activities
GOOGLE_PLACES_API_KEY=your_google_places_key

# Optional — LangSmith tracing (set LANGSMITH_TRACING=false to disable)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=group-travel-agent
```

> **Currency exchange rates** use the free [fawazahmed0 currency API](https://github.com/fawazahmed0/exchange-api) — no key needed.

#### Getting API keys

| Key | Where to get it |
|---|---|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `SEARCH_API_KEY` | [searchapi.io](https://www.searchapi.io) — select Amadeus engine |
| `GOOGLE_PLACES_API_KEY` | [Google Cloud Console](https://console.cloud.google.com) — enable **Places API (New)** |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) (optional) |

### 4. Initialise the database

The database and ChromaDB vector store are created automatically on first run. No manual migration is needed.

---

## Running

```bash
uv run python -m src.cli
```
## Example Session

```
You: I want to plan a group trip. Alice is flying from Dublin and Bob from Lagos.

Agent: What month are you thinking of travelling, and for how many nights?

You: June, about 7 nights

Agent: Got it. Any budget limit per person, or are you open to the best value option?

You: Keep it under $1500 per person

Agent: Here's what I have — please confirm:
  • Alice  — Dublin (DUB)
  • Bob    — Lagos (LOS)
  • June 2025, 7 nights, budget $1500/person

You: yes

Agent: Here's the execution plan:
  Step 1  — Check past searches for cached results
  Step 2  — Find flights for Alice (DUB → IST, LIS, AMS, BCN, MAD)
  Step 3  — Find flights for Bob  (LOS → IST, LIS, AMS, BCN, MAD)
  Step 4  — Find hotels in IST, LIS, AMS, BCN, MAD
  Step 5  — Convert all prices to USD
  Step 6  — Rank destinations by total group cost
  Step 7  — Save results

Approve this plan? (yes/no/feedback)
You: yes

[searches run...]

Top destinations:
  1. Lisbon (LIS)   — $1,840 total  |  Alice: €620  |  Bob: ₦720,000
  2. Istanbul (IST) — $2,100 total  |  Alice: €680  |  Bob: ₦820,000
  3. Madrid (MAD)   — $2,340 total  |  Alice: €720  |  Bob: ₦910,000

Fetch activities for how many destinations? (1 / 2 / 3)
You: 2
```

---

## Configuration Reference

All optional settings have sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `SQLITE_DB_PATH` | `data/travel_agent.db` | Path to SQLite database |
| `CHROMA_DB_PATH` | `data/chroma` | Path to ChromaDB storage |
| `CACHE_TTL_HOURS` | `24` | How long flight/hotel results are cached |
| `VOLATILE_CURRENCY_TTL_HOURS` | `6` | Shorter TTL for volatile currencies (NGN, TRY, ARS) |
| `MAX_TRAVELLERS` | `5` | Maximum group size |
| `MAX_CANDIDATE_DESTINATIONS` | `10` | Max destinations evaluated per search |
| `TOP_ACTIVITIES` | `3` | Activities returned per destination |
| `ACTIVITIES_RADIUS_METRES` | `15000` | Search radius for Google Places (metres) |

---

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src
```

---

## Project Structure

```
group-travel-agent/
├── src/
│   ├── cli.py              # Terminal entry point
│   ├── graph.py            # LangGraph state machine
│   ├── state.py            # Data models
│   ├── agents/             # Individual agent implementations
│   ├── memory/             # Episodic, procedural, semantic memory
│   ├── tools/              # External API clients
│   ├── prompts/            # DSPy prompt modules
│   └── utils/              # LLM config and helpers
├── data/
│   ├── iata.py             # IATA codes, hotel prices, city coordinates
│   └── semantic_seed/      # Destination seed data for ChromaDB
├── tests/
├── .env.example
└── pyproject.toml
```
