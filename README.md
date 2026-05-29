# Group Travel Agent

An AI-powered travel planning assistant that finds the cheapest destination city for a group of people flying from **different origin cities**. Tell it who is travelling and from where — it searches flights, hotels, and activities across candidate destinations and returns a ranked, cost-compared shortlist with full per-traveller breakdowns in local currencies.

---

## Why This Project

Planning a group trip when everyone lives in a different city is genuinely hard. You have to coordinate flights from multiple origins, compare hotels, convert currencies, and do it all repeatedly across a dozen possible destinations before you can even start to compare options. This is exactly the kind of multi-step, multi-source research problem that a multi-agent system is well suited for.

The goal was to build something practically useful while also exploring the full surface area of modern agentic AI: LangGraph state machines, ReWOO-style plan-then-execute reasoning, Human-in-the-Loop interrupts, parallel tool execution, a multi-layer memory system, and DSPy prompt optimisation — all wired together into a single coherent terminal application.

---

## Agent Architecture

The system is a **LangGraph state machine** with specialised agents and three Human-in-the-Loop (HITL) interrupt checkpoints. The full flow:

```
START
  → user_query  (Intent Classifier + Elicitation)
      → OUT_OF_SCOPE: END
      → NEEDS_INFO:   elicitation loop (HITL interrupt per question)
      → READY:        HITL-1 (confirm details)
          → CORRECTED: user_query (loop back)
          → CONFIRMED: plan (Orchestrator generates ReWOO XML)
              → validate_plan → [replan →] HITL-Plan (approve / give feedback)
                  → execute (parallel tool calls follow the plan)
                      → HITL-2 (pick how many destinations to drill into)
                          → activities
                              → END
```

### Nodes / Agents

| Agent | Role |
|---|---|
| **Intent Classifier** | Parses the user's message, extracts travel details, asks clarifying questions for anything missing. Routes to `OUT_OF_SCOPE`, `NEEDS_INFO`, or `READY`. |
| **Orchestrator** | Generates a ReWOO XML execution plan — an ordered, dependency-aware list of steps. Steps without dependencies are flagged for parallel execution. |
| **Traveller Agent × N** | One instance per traveller. Searches live flights from each origin to all candidate destinations in parallel. |
| **Accommodation Agent** | Looks up hotel costs for each candidate destination. |
| **Currency Agent** | Converts all prices to USD for fair comparison, plus gives each traveller their cost in their home currency. |
| **Ranker Agent** | Ranks destinations by total group cost (flights + accommodation). |
| **Activities Agent** | Fetches top-rated tourist attractions (via Google Places) for the user's chosen destinations. |
| **Memory Agent** | Reads from and writes to the memory system before and after searches to avoid redundant API calls. |

### Memory System

```
Episodic   (SQLite)   — recent flight & hotel search results  [24h TTL]
Procedural (SQLite)   — historical winning destinations per origin pattern
Semantic   (ChromaDB) — destination knowledge base + activity cache
```

### External APIs

```
SearchAPI → Amadeus   — live flight search
Google Places API     — tourist activities
fawazahmed0 currency  — exchange rates (free, no key required)
```

### Key Libraries

- **LangGraph** — state machine, checkpointing, HITL interrupts
- **DSPy** — prompt modules for structured LLM outputs
- **ChromaDB** — vector store for semantic memory
- **Rich** — terminal UI
- **Tenacity** — retry logic for flaky API calls
- **LangSmith** — optional tracing

---

## What I Learned

**LangGraph state machines are powerful but require discipline.** The graph's conditional routing, interrupt/resume mechanics, and shared state dict mean the architecture is expressive — but bugs often show up as subtle state-key mismatches that are hard to debug without tracing enabled. LangSmith made this tractable.

**ReWOO separates planning from execution cleanly.** Having the orchestrator emit an XML plan that the execute node follows step-by-step (rather than letting the LLM decide tools on the fly) made the system far more predictable and testable. It also made HITL plan approval natural — the user sees the plan before any API call happens.

**Parallel execution needs explicit dependency modelling.** The plan tracks `depends_on` between steps so independent traveller searches can run simultaneously. Getting this right required careful thought about what data each step actually needs.

**Multi-layer memory is worth the complexity.** Episodic caching alone (avoiding repeat flight searches) cuts API costs significantly in iterative sessions. Semantic memory (ChromaDB) enables fuzzy destination matching and activity caching that a plain key-value store cannot.

**DSPy is better than raw prompt strings for structured outputs.** Replacing hand-written prompts with DSPy modules made it easier to iterate on extraction accuracy for things like parsing traveller details from free-text input.

**HITL interrupt/resume in LangGraph is non-trivial.** The graph pauses mid-execution, serialises state to the checkpointer, and resumes when the CLI sends a `Command(resume=...)`. Getting the CLI to drive this loop — printing the right message, collecting input, and resuming — took several iterations.

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

### 4. Database

The SQLite database and ChromaDB vector store are created automatically on first run. No manual migration is needed.

---

## Running the System

```bash
uv run python -m src.cli
```

Or, if you installed the package scripts:

```bash
uv run travel
```

The agent runs as a multi-turn conversation in your terminal. It will:

1. Ask clarifying questions until it has enough information (origins, dates, budget)
2. Show you a summary and ask you to confirm
3. Generate and display a search plan for your approval
4. Run all searches (flights, hotels, currency conversion, ranking)
5. Show the top destinations and ask which ones to fetch activities for

---

## Example Session

```
You: Two of us — I'm in Lagos and my friend is in Dublin. July, 5 nights, budget $600 per person.

Agent: Here's what I have:
  - Alice: flying from Dublin (DUB)
  - Bob: flying from Lagos (LOS)
  - Month: July
  - Duration: 5 nights
  - Budget: up to $600/person
Is that correct? (yes / no)

You: yes

Agent: Here's my search plan:
  • Check past searches for cached results
  • Simultaneously:
      · Find flights for Alice (DUB → IST, LIS, AMS, BCN, MAD)
      · Find flights for Bob (LOS → IST, LIS, AMS, BCN, MAD)
  • Find hotels in IST, LIS, AMS, BCN, MAD
  • Convert all prices to USD
  • Rank destinations by total group cost
  • Save results for next time
Approve? (yes / no)

You: yes

[searches run in parallel...]

Top destinations:
  #1 Lisbon (LIS) — $1,120 total
       Alice: €420 | Bob: ₦680,000
  #2 Istanbul (IST) — $1,340 total
       Alice: €460 | Bob: ₦820,000
  #3 Madrid (MAD) — $1,580 total
       Alice: €510 | Bob: ₦960,000

Get activities for:
  1 — Just #1
  2 — #1 and #2
  3 — All three
Enter 1, 2, or 3:

You: 2
```

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

## Configuration Reference

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

## Project Structure

```
group-travel-agent/
├── src/
│   ├── cli.py              # Terminal entry point (Rich UI, HITL driver)
│   ├── graph.py            # LangGraph state machine — nodes, edges, routing
│   ├── state.py            # Pydantic data models and state schema
│   ├── agents/
│   │   ├── intent_classifier.py
│   │   ├── orchestrator.py   # ReWOO plan generation, validation, execution
│   │   ├── traveller.py
│   │   ├── accommodation.py
│   │   ├── currency.py
│   │   ├── ranker.py
│   │   ├── activities.py
│   │   └── memory_agent.py
│   ├── memory/
│   │   ├── episodic.py       # SQLite short-term cache
│   │   ├── procedural.py     # SQLite historical patterns
│   │   ├── semantic.py       # ChromaDB vector store
│   │   └── db.py
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

Built by Aghogho Joy Olokpa — [LinkedIn](https://www.linkedin.com/in/aghogho-olokpa-1b0b11115)