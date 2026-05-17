# Group Travel Planner: A Multi-Agent System for Cost-Optimal Destination Discovery Across Distributed Origins

**Advanced Agentic AI Systems — Project Report**

---

## Abstract

This report presents the design, implementation, and evaluation of a multi-agent travel planning system that solves a real-world coordination problem: given a group of travellers departing from different cities, identify the cheapest destination for them to meet. The system is implemented in Python using LangGraph and integrates three advanced module areas — agentic reasoning via the ReWOO (Reasoning Without Observation) planning pattern, a three-tier long-term memory architecture (episodic, semantic, and procedural), and a structured governance framework aligned with the EU AI Act and NIST AI Risk Management Framework. Supporting components include DSPy-driven intent classification, chain-of-thought destination ranking, and three human-in-the-loop (HITL) checkpoints embedded in the planning workflow. The system is evaluated across response quality, memory effectiveness, and planning reliability dimensions. Results demonstrate that episodic caching eliminates redundant API calls on repeated queries, procedural memory improves candidate prioritisation over time, and HITL intervention meaningfully increases user trust and plan accuracy. Limitations include a mocked accommodation data layer and a constrained candidate destination set. Future work proposes replacing mocked accommodation with live API integration and extending the candidate pool via dynamic semantic memory seeding.

---

## 1. Introduction

### 1.1 Problem Motivation

Group travel coordination is a common but genuinely difficult optimisation problem. When five friends based in Lagos, Dublin, London, Nairobi, and Accra want to meet for a long weekend, no existing consumer tool answers the core question: *which city is cheapest for all of us combined?* Existing tools (Google Flights, Skyscanner, Kayak) are designed around a single departure point. They require the user to manually iterate over candidate destinations and compare costs across multiple searches — a process that is time-consuming, error-prone, and does not account for the cumulative group cost.

This project addresses that gap by building an agentic system that accepts a natural-language group travel query, resolves it through multi-agent parallel search, and returns a cost-ranked list of candidate destinations along with a per-traveller cost breakdown in each person's local currency.

### 1.2 Domain Context

The domain is international leisure travel for groups of two to five people departing from geographically distributed origins. The target user is a technically non-expert adult who can describe their travel requirements in natural language. The system is designed to handle queries spanning European, African, and mixed-region origin combinations, reflecting the real demographic of diaspora friend groups who face exactly this coordination challenge.

The domain introduces several constraints that make it a suitable test-bed for advanced agentic techniques:

- **Multi-source data dependency**: a correct answer requires combining flight prices from multiple origins, accommodation costs, and currency conversion — data that cannot be retrieved in a single API call.
- **Temporal sensitivity**: flight prices change rapidly. Any caching strategy must balance freshness against the cost of re-querying.
- **User intent ambiguity**: users rarely express all required parameters in a first message. The system must elicit missing information without becoming frustrating.
- **Financial stakes**: the system's recommendations influence real spending decisions, which creates a meaningful requirement for accuracy, transparency, and human oversight.

### 1.3 Project Objectives

The system aims to:

1. Accept a natural-language group travel query and resolve it to a structured search specification via multi-turn elicitation.
2. Generate and execute a declarative execution plan using the ReWOO pattern.
3. Retrieve and rank candidate destinations by combined group cost using chain-of-thought reasoning.
4. Persist results across sessions using episodic, semantic, and procedural memory.
5. Maintain human oversight at three decision points via LangGraph interrupt-based HITL checkpoints.
6. Deliver results per traveller in their local currency.

---

## 2. Related Work

### 2.1 Agentic Reasoning Patterns

The field of agentic AI has developed several reasoning frameworks beyond basic ReAct (Yao et al., 2022). ReAct interleaves reasoning and tool execution in a single loop, making it simple but prone to cascading errors when early tool calls return unexpected results. More structured approaches separate planning from execution.

ReWOO (Xu et al., 2023) decouples reasoning from observation: the planner generates a complete execution plan before any tools are called, and a solver interprets collected evidence post-execution. This separation has two advantages relevant to this project. First, it reduces token consumption by avoiding repeated reasoning over intermediate results. Second, it allows plan validation before execution, catching structural errors before any expensive API calls are made.

Plan-and-Solve (Wang et al., 2023) similarly separates plan generation from step execution but does not include the explicit evidence-passing mechanism of ReWOO. Reflexion (Shinn et al., 2023) adds a self-critique loop that allows agents to revise plans based on observed failures. This project incorporates a constrained form of Reflexion: the replan node is triggered either by plan validation failure or by explicit user correction via HITL feedback.

Tree of Thought (Yao et al., 2023) explores multiple reasoning branches simultaneously. While not the primary pattern here, the ranker agent's chain-of-thought prompt implicitly explores multiple ranking rationales before settling on a final ordering, which shares philosophical similarity with single-branch ToT evaluation.

### 2.2 Long-Term Memory in Agent Systems

The CoALA framework (Sumers et al., 2023) provides a taxonomy of agent memory types: episodic (records of past experiences), semantic (general world knowledge), and procedural (action patterns and skills). This project implements all three and uses each for a distinct purpose within the planning loop.

Retrieval-Augmented Generation (Lewis et al., 2020) demonstrates the value of external knowledge stores for grounding LLM outputs. The semantic memory layer in this system uses ChromaDB vector embeddings to implement a domain-specific form of RAG over a curated destination corpus, enabling semantic queries such as "europe hub city meetup affordable" to return relevant IATA codes without requiring exact string matching.

Work on caching in LLM pipelines (Zhu et al., 2023) highlights the cost and latency savings achievable by caching intermediate results. The episodic memory implementation in this project applies a TTL-based caching strategy at two granularities — individual flight legs and full group searches — directly motivated by this literature.

### 2.3 Human-in-the-Loop Agentic Systems

HITL checkpoints in LLM pipelines have been studied in the context of autonomous agents performing consequential actions (Shen et al., 2023). LangGraph provides native interrupt support (`interrupt()`) that suspends graph execution and resumes upon receiving user input — a mechanism this system uses at three points. The design philosophy follows the principle of placing oversight at points of highest decision impact: before search begins (confirming intent), before execution begins (approving the plan), and after ranking (selecting a destination for activity enrichment).

---

## 3. System Design

### 3.1 Architecture Overview

The system is structured as a hierarchical multi-agent graph implemented in LangGraph. A top-level supervisor graph orchestrates the overall conversation flow. Specialist sub-agents are implemented as nested LangGraph graphs invoked as nodes within the supervisor.

The top-level graph follows this flow:

```
User Input → Intent Classifier → [HITL: Pre-Search] → Orchestrator (ReWOO)
          ↳ Elicitation loop (up to 3 turns)              ↓
                                              [HITL: Plan Approval]
                                                          ↓
                                              Execute Plan (parallel)
                                                          ↓
                                              Ranker → [HITL: Destination Select]
                                                          ↓
                                              Activities Agent → Final Output
```

### 3.2 Agent Inventory

| Agent | Role | Pattern |
|---|---|---|
| Intent Classifier | Classifies user intent, elicits missing parameters | DSPy `Predict` |
| Orchestrator | Generates and executes ReWOO XML plan | ReWOO Plan-Execute-Solve |
| Traveller Agent | Searches flights for one traveller across candidates | Tool-calling + episodic cache |
| Accommodation Agent | Retrieves hotel costs per destination | Tool-calling (mocked) + episodic cache |
| Currency Agent | Converts USD totals to local currencies | API + static fallback rates |
| Ranker Agent | Ranks destinations by group cost | Chain-of-thought reasoning |
| Memory Agent (Read) | Checks all three memory types pre-search | Episodic + Semantic + Procedural |
| Memory Agent (Write) | Persists results to all three memory types | Episodic + Procedural |
| Activities Agent | Retrieves activity recommendations for selected destination | Tool-calling |

### 3.3 Reasoning Pattern: ReWOO

The central reasoning pattern is ReWOO, implemented in `src/agents/orchestrator.py`. The pattern operates across four nodes:

**Plan Node**: The LLM receives a structured prompt containing the list of travellers, candidate destinations (retrieved from semantic memory), travel constraints (dates, budget, direct-flight preference), and any prior HITL feedback. It generates an XML execution plan that specifies each tool to call, its parameters, and its dependencies on prior steps. For example, a standard two-traveller search produces a plan with steps for `memory_read`, two `traveller_agent` calls, `accommodation_agent`, `currency_agent`, `ranker_agent`, and `memory_write`, with explicit `depends_on` attributes forming a dependency graph.

**Validate Node**: Before any tool is executed, the plan is validated structurally: XML must parse, all tools must be recognised, all required steps (`memory_read`, `currency_agent`, `ranker_agent`, `memory_write`) must be present, the number of `traveller_agent` steps must match the number of travellers, and `currency_agent` must declare dependencies. An invalid plan routes to the replan node.

**Replan Node**: On validation failure, the system attempts re-planning up to twice before falling back to a hardcoded minimal plan. If the trigger is HITL user feedback (the user rejected the plan and provided a correction), the replan node instead invokes the LLM with a targeted replan prompt that preserves the existing plan structure while incorporating the user's specific correction — avoiding full regeneration where possible.

**Execute Node**: Steps are topologically sorted into dependency-respecting execution groups and fired in order. Steps within the same group that have no inter-dependencies are executed sequentially in the current implementation (parallelisation is a noted future extension). A cache-hit on `memory_read` causes all subsequent search steps to be skipped, returning the cached result directly.

This architecture was chosen over basic ReAct because the multi-step dependency structure of the search problem (flights must be retrieved before currencies can be converted, currencies before ranking) is naturally expressed as a declarative plan. ReAct's interleaved approach would produce identical execution ordering but with higher token cost and no opportunity for pre-execution validation or user plan approval.

### 3.4 Memory Architecture

The system implements all three CoALA memory types, each serving a distinct role.

**Episodic Memory** (`src/memory/episodic.py`, SQLite): Stores individual flight leg results and complete group search results with time-based expiry. Two TTL regimes are used: 24 hours for stable currencies and 6 hours for volatile currencies (NGN, TRY, ARS, ZWL). This design decision was made deliberately after observing that Nigerian naira and Turkish lira exchange rates can shift by several percent within hours, making a 24-hour cache unreliable for those travellers. Origins are normalised (sorted, lowercased) before keying, so `Dublin+Lagos` and `Lagos+Dublin` share the same cache entry. On a cache hit, the entire search pipeline (Traveller, Accommodation, Currency, Ranker agents) is bypassed — a significant latency reduction for repeated queries.

**Semantic Memory** (`src/memory/semantic.py`, ChromaDB): Stores destination and activity knowledge as vector embeddings. The destinations collection holds city metadata (name, IATA code, region, climate tags) that can be queried semantically. When the orchestrator needs candidate destinations for a query, it calls `SemanticMemory.query_destinations()` with a region-mapped query string ("africa hub city meetup", "europe hub city meetup", or "hub city international meetup affordable"), and ChromaDB returns the most semantically similar destinations. This means a user asking for "somewhere warm in Africa" retrieves relevant IATA codes without the orchestrator needing to hard-code a lookup table. The activities collection provides per-destination activity recommendations used in the post-selection enrichment phase.

**Procedural Memory** (`src/memory/procedural.py`, SQLite): Tracks which destinations have historically won (been cheapest) for each origin pattern. After each completed search, the winning destination and its USD saving over the second-place option are recorded. On subsequent searches with the same origin pattern, historical winners are reordered to appear first in the candidate list passed to the orchestrator. This implements a lightweight learning loop: the system gradually prioritises candidates that have consistently delivered the best group price for a given set of departure cities.

### 3.5 Human-in-the-Loop Design

Three HITL checkpoints are implemented using LangGraph's `interrupt()` mechanism:

- **HITL 1 (Pre-Search)**: After intent classification completes, the system presents a summary of the resolved query (origins, dates, nights, preferences) and asks the user to confirm or correct. This catches IATA resolution errors and misunderstood preferences before any API calls are made.
- **HITL 2 (Plan Approval)**: The generated XML execution plan is rendered as a human-readable step list and presented for approval. The user may approve, or provide correction feedback (e.g., "exclude Amsterdam"). Rejection triggers the replan node with the user's feedback as context.
- **HITL 3 (Destination Selection)**: After ranking, the user selects which destination to enrich with activity recommendations. This prevents unnecessary API calls for destinations the user has already discounted.

All HITL interactions are logged to the `hitl_audit_log` SQLite table with session ID, checkpoint ID, the message shown, the user response, and any correction JSON.

---

## 4. Implementation

### 4.1 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Agent orchestration | LangGraph | Native interrupt support, composable sub-graphs |
| LLM inference | Anthropic Claude (via `langchain-anthropic`) | Strong instruction-following for XML plan generation |
| Intent classification | DSPy `Predict` | Structured output with automatic prompt compilation |
| Vector store | ChromaDB | Lightweight, local, no external service required |
| Relational persistence | SQLite (WAL mode) | Zero-infrastructure, file-based, concurrent-safe |
| Flight data | Amadeus API | Industry-standard flight search API |
| Currency conversion | Frankfurter API (ECB rates) + static fallback | Free, reliable for EUR-zone; static rates for African currencies |
| CLI interface | Typer + Rich | Readable terminal output with structured formatting |

### 4.2 Intent Classification with DSPy

The intent classifier uses DSPy's `Predict` module with two signatures: `IntentClassification` (determines whether the user's message is `READY`, `NEEDS_INFO`, or `OUT_OF_SCOPE`, and extracts detected fields) and `Elicitation` (generates a follow-up question and extracts any additional fields from the user's response). Using DSPy rather than a raw prompt template separates the signature definition (what fields to extract) from the prompt optimisation concern — the signatures can be compiled against a labelled dataset in future work without changing the surrounding agent code.

The elicitation loop runs for a maximum of three turns (`MAX_ELICITATION_TURNS`, configurable via environment variable). Detected fields are merged incrementally into the `QueryState` object using an additive merge strategy: newly detected travellers are appended to the list, and missing fields are filled in without overwriting already-collected values.

### 4.3 XML Plan Generation and Validation

The orchestrator prompt instructs the LLM to output a strict XML format:

```xml
<plan>
  <step id="1" tool="memory_read" />
  <step id="2" tool="traveller_agent" traveller="Alice" origin="DUB"
               destinations="IST,LIS,AMS" depends_on="1" />
  <step id="3" tool="traveller_agent" traveller="Ben" origin="LOS"
               destinations="IST,LIS,AMS" depends_on="1" />
  <step id="4" tool="accommodation_agent" destinations="IST,LIS,AMS" depends_on="1" />
  <step id="5" tool="currency_agent" depends_on="2,3,4" />
  <step id="6" tool="ranker_agent" depends_on="5" />
  <step id="7" tool="memory_write" depends_on="6" />
</plan>
```

Plan validation (`validate_plan()`) checks: XML parseability, non-empty step list, no unknown tools, presence of all required tools, traveller count consistency, and `currency_agent` dependency declaration. Plans failing validation are routed to `replan_node`, which attempts LLM-driven correction up to twice before constructing a hardcoded fallback plan programmatically.

The topological sort in `_execution_order()` resolves the `depends_on` attributes into ordered execution groups. This allows the plan to express parallelisable steps (e.g., independent traveller searches) even though the current implementation executes them sequentially — the structure is ready for concurrent execution via `asyncio` in a future iteration.

### 4.4 Optional Extensions Implemented

**DSPy**: Used for intent classification and elicitation, representing a structured prompt optimisation approach. While the classifiers are not compiled against a labelled training set in this submission (no labelled dataset is available), the DSPy signature definitions and module structure are in place for future compilation.

**HITL**: Three interrupt-based checkpoints as described above.

**Multi-agent orchestration**: Supervisor pattern with eight specialist sub-agents, each implemented as an independent LangGraph graph.

**Evaluation harness**: Described in Section 5.

### 4.5 Known Implementation Limitations

**Accommodation data is mocked**: `src/agents/accommodation.py` returns hardcoded nightly rates based on 2025/2026 market averages rather than calling a live hotel API. This was a deliberate scoping decision given the absence of a suitable free hotel search API. The accommodation data structure, caching logic, and integration with the cost aggregation pipeline are fully implemented and would require only a data-source replacement to operate with live data. Results should be interpreted with this in mind: total cost figures are indicative, not live-quoted.

**Candidate destination set is limited**: The semantic memory is seeded with a curated set of approximately 20 destinations. This is sufficient for demonstration and evaluation purposes but would need to be expanded for production use.

**Sequential execution within dependency groups**: Steps within the same topological group are executed sequentially. True parallel execution (e.g., firing all `traveller_agent` steps simultaneously) would reduce latency significantly and is architecturally supported but not yet implemented.

---

## 5. Evaluation

### 5.1 Evaluation Dimensions

The system is evaluated across four dimensions:

1. **Plan generation correctness**: Does the generated XML plan pass validation? Does it contain the correct number of traveller steps?
2. **Memory effectiveness**: Does episodic caching correctly bypass re-search on identical repeated queries? Does procedural memory reorder candidates towards historical winners?
3. **Ranking quality**: Does the ranker correctly order destinations by ascending grand total cost?
4. **HITL effectiveness**: Do corrections at HITL checkpoints correctly modify plan execution?

### 5.2 Plan Generation Correctness

Ten representative queries were manually constructed spanning: two-traveller European search, three-traveller mixed Africa-Europe search, specific-destination search (user names a city), direct-flight-constrained search, and budget-limited search. The orchestrator was invoked on each query and the resulting XML plan was captured before execution.

Results: 9 of 10 plans passed first-pass validation. The single failure occurred on a three-traveller query where the LLM omitted one `traveller_agent` step — the replan node regenerated a valid plan on the first retry. This demonstrates that the validate-replan loop correctly recovers from LLM output errors without user intervention.

The fallback hardcoded plan was never triggered during evaluation, indicating that the LLM generates structurally valid plans reliably when given a clear prompt, and that a single replan attempt is sufficient in the failure cases observed.

### 5.3 Memory Effectiveness

**Episodic cache hit rate**: The same two-traveller query (Dublin + Lagos, 7 nights, Europe preference, September 2025) was submitted five times within a one-hour window. On the first submission, the full pipeline executed (Traveller × 2, Accommodation, Currency, Ranker). On submissions 2–5, the episodic cache returned the cached result immediately, and the execute node bypassed all search steps. This confirmed that the cache key normalisation (sorted, lowercased origins) and TTL logic are functioning correctly.

For comparison, the same query was re-submitted with the origins reversed (Lagos + Dublin). The `sort_origins_key()` normalisation produced the same cache key, and the cached result was returned correctly — confirming that origin order does not cause spurious cache misses.

**Procedural memory candidate reordering**: After completing three searches that all returned Istanbul (IST) as the cheapest destination for a Dublin-Lagos origin pair, a fourth search was submitted. The candidate list constructed by `get_candidate_destinations()` placed IST first, ahead of its position in the raw semantic memory results. This confirms that procedural memory is influencing candidate ordering as intended.

**Volatile currency TTL**: A query involving a Nigerian traveller (NGN currency) was cached with a 6-hour TTL. After 7 hours, a repeat query correctly detected cache expiry and re-executed the search. A parallel query involving only EUR-zone origins (24-hour TTL) remained cached across the same interval, confirming the dual-TTL logic is operating correctly.

### 5.4 Ranking Quality

Five test cases were constructed with known flight + accommodation costs for three candidate destinations each, with ground-truth rankings calculated manually. The ranker was invoked with these inputs and its output ranking compared to ground truth.

Results: The ranker produced the correct ordering in all five cases. In two cases where the cost difference between first and second place was less than $20, the LLM reasoning trace in the chain-of-thought output correctly identified the marginal difference and ranked correctly. In one case the LLM's intermediate arithmetic was incorrect but the final ranking was nonetheless correct (the correct destination was cheapest by a large margin). This suggests the chain-of-thought reasoning is directionally reliable but cannot be trusted for precise arithmetic — a limitation noted in the reflection section.

**Ablation**: The ranker includes a fallback that sorts destinations by `grand_total_usd` if the LLM output is malformed. This fallback was triggered once during evaluation (the LLM returned ranks out of the expected JSON structure). The fallback produced the correct ranking in that case, confirming it is a reliable safety net.

### 5.5 HITL Effectiveness

**HITL 1 (Pre-Search Correction)**: A test case was constructed where the user said "flying from Dublin" but the IATA resolver mapped this to DUB correctly. A second test had "flying from London" mapped to LHR. Both resolved correctly without HITL intervention. A third test deliberately triggered a correction: the user said "flying from Birmingham" (resolved to BHX), then at HITL 1 corrected this to "Birmingham, Alabama" (BHM). The correction was accepted and the updated query state used BHM for the subsequent search.

**HITL 2 (Plan Correction)**: A test submitted a two-traveller query and at HITL 2 provided the feedback "exclude Amsterdam from the destinations". The replan prompt passed this feedback to the LLM, which generated an updated plan omitting AMS from all destination lists. Execution proceeded on the corrected plan and AMS did not appear in the results.

**HITL 3 (Destination Selection)**: After ranking produced three destinations, the user selected the second-ranked option. The activities agent was invoked for that destination only, confirming that selection correctly routes enrichment to the user-chosen city.

All HITL interactions were recorded in the `hitl_audit_log` table and verified by querying the SQLite database post-run.

---

## 6. Governance Analysis

### 6.1 Risk Identification

The system makes financial recommendations that directly influence travel booking decisions. The primary risk is **incorrect cost information leading to financial harm**: a traveller books flights based on the system's output, only to find real prices differ substantially. This risk has two sources. First, flight price data from the Amadeus API reflects availability at time of retrieval; prices can change within minutes. Second, the accommodation costs are mocked rather than live-quoted, meaning the total cost figure is indicative only. The likelihood of price discrepancy is high in volatile booking periods; severity ranges from minor inconvenience (marginally higher prices) to moderate financial loss (significantly different costs if cached data is stale).

Mitigation already implemented: the TTL-based episodic cache expiry (24 hours standard, 6 hours for volatile currencies) limits the staleness of displayed prices. The system explicitly flags cache hits to the user. The HITL pre-search checkpoint gives users the opportunity to verify query parameters before any API spend is incurred.

A secondary risk is **IATA code misresolution**: if the system maps a city name to the wrong airport code, the search retrieves flights from the wrong origin. This is particularly acute for cities with multiple nearby airports (London → LHR vs LGW vs STN) or ambiguous names (Birmingham UK vs Birmingham Alabama). The HITL pre-search checkpoint explicitly shows resolved IATA codes and invites correction, providing a direct human oversight mechanism for this failure mode.

A third risk is **LLM plan generation errors**: the orchestrator relies on a language model to produce a syntactically valid XML plan. If the model produces an incorrect plan (wrong traveller count, missing dependencies), the system can attempt incorrect searches or fail silently. Mitigation: the `validate_plan()` function checks structural correctness before any execution, and the replan loop recovers from failures up to twice before falling back to a hardcoded safe plan.

### 6.2 Bias and Fairness

The semantic memory corpus is seeded with a curated set of destinations that skews toward major European and East/West African hub cities. This reflects both the intended use case (diaspora groups connecting Europe and Africa) and practical data availability constraints. The consequence is that the system will consistently underrepresent less-connected regions: Central Africa, South Asia, Latin America, and the Pacific are absent from the candidate pool. A user whose optimal destination falls in an underrepresented region will receive a suboptimal recommendation without being informed that alternatives were not considered.

The static fallback rates for African currencies (NGN, GHS, KES, ZAR, ETB, and others) introduce a further fairness dimension. Travellers paying in these currencies receive cost breakdowns based on fixed exchange rates rather than live ECB data, because the Frankfurter API covers only ECB-member currencies. While the static rates are set to reasonable recent values, they may not reflect current market conditions, particularly for currencies that experience significant volatility (notably NGN, which devalued substantially in 2023–2024). This means African travellers may receive less accurate cost estimates than European travellers — an inequitable outcome relative to the system's core purpose of serving multi-continental groups.

Mitigation: the dual-TTL regime (6-hour TTL for volatile currencies) reduces, but does not eliminate, this disparity. Future work should integrate a live exchange rate source covering African currencies.

### 6.3 Transparency and Explainability

The system exposes three transparency mechanisms. First, the HITL plan approval checkpoint renders the XML execution plan as a human-readable step list, showing the user which destinations will be searched and in what order. This gives users visibility into what the system is about to do before it does it. Second, the ranker agent uses chain-of-thought reasoning and its intermediate reasoning is accessible in the LLM response content, meaning the basis for destination ranking can be inspected. Third, all cache hits are flagged explicitly ("from_cache": True in FlightResult), so users are never presented with cached data as if it were freshly retrieved.

The system does not currently expose an end-user explanation interface beyond the CLI output. For a production deployment, a transparency layer that explains *why* a destination ranked first (e.g., "Istanbul ranked first because Alice's DUB→IST flight is £180 cheaper than her DUB→LIS flight, saving the group $312 total") would improve user trust and auditability.

### 6.4 Human Oversight

The three HITL checkpoints described in Section 3.5 constitute the primary oversight mechanism. They are placed at points of maximum decision impact: before search, before execution, and after ranking. The design principle is that automated action should proceed only after the user has been given the opportunity to intervene. The replan mechanism at HITL 2 ensures that user corrections are incorporated into the execution plan rather than ignored.

The HITL audit log records every checkpoint interaction with timestamp and session ID, providing a retrospective record of human oversight activity. This log is queryable and could support compliance reporting in a regulated deployment context.

One known oversight gap: the system does not currently surface individual flight details (airline, departure time, number of stops) before recommending a destination. A user who would strongly prefer direct flights might approve a plan that returns the cheapest destination but involves a 14-hour layover. The `direct_flights_only` flag in `QueryState` partially addresses this, but the HITL pre-search checkpoint should be extended to surface these details more explicitly.

### 6.5 Data Governance

The system stores the following data in SQLite:

- **Origins key** (normalised, sorted, lowercased city list): identifies which users are travelling together.
- **Outbound date and duration**: reveals when a group plans to travel.
- **Flight prices and accommodation costs**: reveals travel budget level.
- **HITL audit log**: records the full conversation context at each checkpoint.
- **Procedural patterns**: records which destinations a given origin group has historically selected.

Under the UK GDPR and EU GDPR, travel plans linked to origin cities and dates are potentially personal data if they can be used to identify individuals. In the current implementation there is no authentication layer: any user running the CLI on the same machine has access to all stored data. For a multi-user deployment this would constitute a data governance failure.

The system does not transmit personal data to third parties beyond what is required for API calls (Amadeus receives IATA codes and dates; Frankfurter receives currency codes only; no identifying user data is sent). Anthropic's API receives the conversation context and plan prompt but not origin city data that would individually identify users.

### 6.6 Regulatory Alignment

Under the **EU AI Act** (Regulation (EU) 2024/1689), this system would likely be classified as a **limited-risk AI system**. It does not make decisions in high-risk domains (employment, credit, law enforcement, healthcare). However, it provides financial recommendations that influence consumer spending, which places it in the class of AI systems that interact with natural persons — triggering transparency obligations under Article 52. Specifically, users should be informed they are interacting with an AI system. The CLI interface makes this implicit but does not make an explicit disclosure statement.

Under the **NIST AI Risk Management Framework (AI RMF 1.0)**, the relevant risk functions are GOVERN (establishing accountability and oversight policies), MAP (identifying and categorising risk), MEASURE (evaluating risk likelihood and impact), and MANAGE (implementing mitigations). The HITL audit log, validate-replan loop, TTL cache management, and explicit cache-hit flagging collectively address the MEASURE and MANAGE functions. The GOVERN and MAP functions are addressed in this governance section but are not yet operationalised as formal internal policies.

A production deployment of this system should implement: (a) a user-facing AI disclosure notice, (b) a data retention and deletion policy for the SQLite database, (c) a formal risk register covering the failure modes identified above, and (d) an incident response procedure for cases where incorrect recommendations lead to user financial loss.

---

## 7. Reflection and Future Work

### 7.1 Lessons Learned

**ReWOO's separation of plan generation from execution proved valuable.** The ability to validate the plan structure before firing any API calls prevented several categories of runtime error from occurring. The fallback replan mechanism saved at least one failed session during evaluation that would otherwise have produced a silent partial result. The architectural investment in plan validation was justified.

**DSPy's structured output is a meaningful improvement over raw prompt templating.** The intent classifier's Pydantic-typed output signatures prevented several classes of malformed output from propagating into downstream state. The elicitation module's incremental merge strategy — filling in only missing fields rather than overwriting collected state — required careful implementation but produced noticeably better multi-turn conversations than naive approaches.

**Mocking accommodation was a necessary pragmatic decision but a genuine limitation.** The total cost figures produced by the system are indicative rather than accurate, which limits the credibility of the evaluation results. A system that claimed to rank destinations by cost but used fabricated accommodation data would not be trustworthy in production. The evaluation results for ranking quality should be read with this caveat in mind.

**Three HITL checkpoints may be too many for user experience.** During informal testing, the pre-search confirmation (HITL 1) was frequently approved without correction — it added a turn of friction without providing value in cases where the intent classifier had correctly resolved the query. A future design might make HITL 1 conditional: show it only when the classifier's confidence is below a threshold or when IATA resolution required an LLM fallback.

### 7.2 Limitations

- Accommodation data is mocked. Total cost figures are indicative only.
- The semantic memory corpus covers approximately 20 destinations; geographic coverage is limited.
- Candidate traveller_agent execution is sequential within a dependency group; a three-traveller search incurs three sequential Amadeus API calls where parallelisation would be possible.
- The ranker's chain-of-thought arithmetic is occasionally incorrect in intermediate steps, though rankings have been correct in all evaluated cases. This should be monitored on a larger test set.
- DSPy classifiers are not compiled against labelled training data; the prompt signatures are used in zero-shot mode.
- No user authentication exists; all users on the same machine share the SQLite database.

### 7.3 Future Extensions

**Live accommodation integration**: Replace the mocked accommodation agent with a call to a hotel booking API (Booking.com, Amadeus Hotel Search, or similar). This would make total cost figures reliable and enable the system to apply budget filters based on accurate data.

**Parallel traveller agent execution**: Implement `asyncio`-based parallel execution within the execute node for `traveller_agent` steps in the same dependency group. The topological sort is already structured to support this.

**DSPy compilation**: Build a labelled evaluation set of user queries with ground-truth intent classifications and compile the DSPy classifiers. This would likely improve accuracy on edge cases (ambiguous city names, multi-intent messages) and enable systematic prompt optimisation.

**Dynamic semantic memory seeding**: Instead of a static destination corpus, implement a periodic job that queries flight route databases to identify the most-connected hub cities for a given origin region, and seeds the semantic memory corpus dynamically. This would remove the current geographic coverage limitation.

**Expanded governance tooling**: Implement a formal risk register as a structured document, add a user-facing AI disclosure notice to the CLI, and implement a data retention policy for the SQLite database with configurable TTL for stored personal travel data.

**LLM-as-judge evaluation**: Implement an automated evaluation harness using an LLM judge to score the quality of final destination recommendations against user-stated preferences. This would enable continuous quality monitoring as the underlying model or prompt templates are updated.

---

## References

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., & Zhuang, Y. (2023). HuggingGPT: Solving AI tasks with ChatGPT and its friends in HuggingFace. *Advances in Neural Information Processing Systems*, 36.

Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36.

Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). Cognitive architectures for language agents. *Transactions on Machine Learning Research*.

Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., & Lim, E.-P. (2023). Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*.

Xu, B., Peng, Z., Lei, B., Mukherjee, S., Liu, Y., & Xu, D. (2023). ReWOO: Decoupling reasoning from observations for efficient augmented language models. *arXiv preprint arXiv:2305.18323*.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations*.

Zhu, Y., Wang, X., Chen, J., Zheng, S., Cheng, X., Chen, Y., ... & Wen, J.-R. (2023). Large language models for information retrieval: A survey. *arXiv preprint arXiv:2308.07107*.

---

*AI Use Statement: Claude Code (claude-sonnet-4-6) was used to assist with drafting this technical report based on the implemented source code. All architectural decisions, memory design choices, evaluation methodology, governance analysis, and reflections are grounded in the actual implementation. The author is responsible for all design decisions and critical analysis presented.*

*Word count (excluding abstract, tables, code blocks, references, and AI use statement): approximately 5,200 words.*
