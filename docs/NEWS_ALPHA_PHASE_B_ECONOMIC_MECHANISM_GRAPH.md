# News Alpha Portfolio Agent — Phase B: Economic Mechanism Graph

Status: **ACCEPTED.** Signal Path Discovery and the mechanism-adjusted impact landed in Phase C (`docs/NEWS_ALPHA_PHASE_C_SIGNAL_PATH_DISCOVERY.md`).

Phase B extends the same pipeline by one stage:

```
ResearchMandate → Allowed Asset Universe → News/Event → Initial Impact Scan → Economic Mechanism Graph
   (Phase A)            (Phase A)            (existing)        (Phase A)              (Phase B, new)
```

Run it:

```
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py                  # offline vertical slice
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached --scripted-proposal
streamlit run python/alpha_agent/ui/app.py                                   # Agent → News Impact Triage → a card
```

## 1. Reuse audit: what "mechanism" already meant

| Where | What "mechanism" means there | Plane |
|---|---|---|
| `knowledge.models.EconomicMechanism` | A closed catalog of **quant** mechanisms (TREND, MOMENTUM, CARRY, TERM_STRUCTURE, BREAKOUT …). Despite the name, these are price patterns a strategy family exploits. | Evidence / strategy |
| `core.mechanism` | A re-export of that same enum (Phase 5 asset-neutral spine). | — |
| `translation.mechanism_library` | `NewsCategory` → quant-mechanism templates. `causal_chain` is prose about the *market reaction* ("surprise → positioning → trend"). | Hypothesis → factor |
| `alpha_graph` | `MechanismGraphView` / `build_mechanism_graph`: quant mechanism → instruments → registry evidence. | Evidence |
| `news_alpha.channels` (Phase A) | `EconomicChannel`: the first-order route news takes into markets. | Hypothesis (triage) |

Nothing owned **economic transmission** (AI capex ↑ → compute demand ↑ → HBM demand ↑). So Phase B:

- **Does not overload `EconomicMechanism`.** A test asserts the new vocabularies are disjoint from it.
- **Does not name a type `MechanismGraph`.** `alpha_graph` already has `MechanismGraphView` / `build_mechanism_graph` for the quant evidence plane. The typed graph is `TransmissionGraph`; the per-event stage output is `EconomicMechanismGraph`.
- **Roots the graph at Phase A's `EconomicChannel`.** That is the closest existing concept, and each channel's direction shift sets the root's movement.
- **Lives in `news_alpha`, next to the stage it extends.** `core` holds re-exports only and `translation` owns the quant chain, so neither could own this without a second meaning for "mechanism".

## 2. Two planes, kept apart

| Hypothesis plane (this phase) | Evidence plane (unchanged) |
|---|---|
| News/Event → economic transmission → consequences → (later) signal paths | `alpha_graph`: quant mechanism → instrument → experiment → evidence |
| `news_alpha.transmission`, `.transmission_library`, `.mechanism_graph` | `alpha_graph`, `alpha_memory`, registry |

Separation is test-enforced both ways. The Phase B modules import nothing from `alpha_graph`, `alpha_memory`, `registry`, `knowledge`, `translation`, `validation` or `agents`, and none of those import `news_alpha`. `EconomicMechanismGraph.plane` is the literal `"HYPOTHESIS"`; relabelling it fails validation.

## 3. The typed model (`news_alpha/transmission.py`)

- **`EconomicState`**: a node, meaning an economic quantity such as electricity demand or the policy-rate path. It has **no symbol, root, ticker or asset-domain field** (test-enforced), so the tradable universe cannot constrain it.
- **`TransmissionClaim`**: one asserted relationship, as supplied. Fields: `source`, `target`, `channel` (`TransmissionChannel`), `polarity` (POSITIVE/NEGATIVE/AMBIGUOUS/UNKNOWN), `lag` (IMMEDIATE…YEARS/UNKNOWN), `confidence` (HIGH/MEDIUM/LOW/UNKNOWN), `origin` and `provenance`. **It has no status field.**
- **`EdgeOrigin`**: who introduced the claim.
  - `OBSERVED`
  - `EXTERNALLY_SOURCED`
  - `MODEL_PROPOSED`
  - `MANUALLY_SEEDED`
- **`ProvenanceSource`**: a typed source with fields `kind`, `reference`, `title`, `publisher`, `published`, the verbatim `claim`, `verification` and `verified_on`.
- **`TransmissionEdge`**: the graph-resolved link. All duplicate claims are merged in and preserved, and `support` is **computed**, never supplied.
- **`TransmissionGraph`**: holds the states, edges, feedback loops, safety issues and rejected claims, plus a content fingerprint.

`TransmissionLag` is deliberately separate from Phase A's `ImpactHorizon`. Prices can react at once to demand that arrives years later.

## 4. Provenance rule (`support-status/1`)

| Status | When |
|---|---|
| **UNRESOLVED** | An opposite-sign link exists on the same (source, target), **or** no claim carries the provenance its origin requires. Seeds need a `PLATFORM_RULE`, model proposals a `MODEL_OUTPUT`, external claims an `EXTERNAL_DOCUMENT`, and observed claims `OBSERVED_DATA`. |
| **SUPPORTED** | At least one claim cites a source whose recorded statement was **verified against the source itself**: an external document, observed data, or a definitional identity. |
| **PROPOSED** | Everything else: reasoned or proposed, not yet backed. |

How a model is kept from promoting itself:

- A `MODEL_OUTPUT` or `PLATFORM_RULE` source can never be `VERIFIED` (validator).
- `TransmissionProposal` / `ProposedLink` / `ProposedCitation` are the only shape an LLM's output may take. They have no status, origin or verification field (`extra="forbid"`), so an output that tries to mark its own link SUPPORTED fails validation.
- `claims_from_proposal` records every model citation as `UNVERIFIED`.

The only promotion path is an independent step, human or deterministic, that verifies a source and attaches it. A test demonstrates it. No LLM is wired to this schema yet (see §11).

## 5. Graph safety

| Situation | Behaviour |
|---|---|
| Duplicate nodes | Resolved by id, label or alias, ignoring case and punctuation ("HBM Demand" = "high bandwidth memory demand" = `hbm_demand`). A conflicting redeclaration or an ambiguous alias is a `STATE_DEFINITION_CONFLICT`; the first declaration wins, visibly. |
| Undeclared node | Admitted as `UNSPECIFIED` (dashed on the map) plus `UNDECLARED_STATE`. |
| Duplicate edges | Merged on (source, target, sign), with every claim and source preserved (`DUPLICATE_MERGED`). Confidence becomes the most conservative stated. |
| Lag disagreement | The link's lag becomes UNKNOWN plus `LAG_DISAGREEMENT`; each claim keeps its own lag. |
| Conflicting directions | Both links are kept and both are UNRESOLVED (`DIRECTION_CONFLICT`). Downstream states become MIXED. Nothing is averaged. |
| Cycles | Every simple cycle up to 8 states is reported as REINFORCING, BALANCING or INDETERMINATE. Propagation walks simple paths only, so it always terminates. Self-loops are rejected but preserved. |
| Unknown horizon / confidence / sign | `UNKNOWN_LAG`, `UNKNOWN_CONFIDENCE` or `UNSIGNED_LINK`. A chain is never faster or surer than its weakest link. |
| Missing provenance | UNRESOLVED plus `MISSING_PROVENANCE`. |
| Unverified citation | `UNVERIFIED_CITATION` (the link stays PROPOSED). |
| Search bounds | `HOP_LIMIT_REACHED` (default 6 hops), `LOOP_SEARCH_TRUNCATED`, `PATH_SEARCH_TRUNCATED`. |

## 6. Stage logic (`news_alpha/mechanism_graph.py`)

1. **Anchor.** Each detected Phase A channel with an `AnchorRule` anchors its root state. The channel's shift sets the direction; a missing or conflicting cue sets UNKNOWN. Two channels moving one root in opposite directions produce `ANCHOR_CONFLICT`. An unseeded channel is listed in `unseeded_channels`, never dropped silently.
2. **Expand.** The event's graph is whatever part of the claim pool is reachable from the anchors within `max_hops`. The pool is the seed library plus any `extra_claims`, such as a model proposal. It is not a hard-coded subgraph; a custom `TransmissionLibrary` produces a different graph.
3. **Propagate (`sign-propagation/1`).** Along every simple path, implied movement = anchor movement × product of link signs. Paths that disagree make a state **MIXED**, and both paths are kept with their slowest lag and weakest link. With no direction cue, states are shown **relative** to the anchor (+ moves with it, − against it).

**The mandate never shapes the graph.** The builder reads only `scan.event` and `scan.channels`. Tests assert an identical fingerprint under four different mandates, and again with the scan's assessments removed. An Equity-only mandate still sees electricity demand, copper demand and the policy-rate path.

## 7. First domain: AI infrastructure (plus two compact seeds)

The AI slice has 15 states and 19 links. It covers both prompt chains and the spender side that Phase A deferred ("AI capex carries no sign … the Mechanism Graph resolves that"):

- AI infrastructure investment ↑ → compute ↑ → accelerators ↑ → HBM ↑ → foundry & memory capacity expansion ↑ → semiconductor equipment ↑
- AI infrastructure investment ↑ → data-center construction ↑ → electricity demand ↑ → grid investment ↑ → power equipment ↑ (plus gas-for-power and copper)
- **AI spenders' free cash flow ⇅.** It moves DOWN within months (capex is a cost) and UP over years (AI services revenue, low confidence). This is resolved into a visible disagreement, not a single sign.
- **Leading-edge foundry utilization ⇅** is UP within months. It is DOWN over years because new capacity relieves it, which forms a **balancing loop**.

**Verified sources (opened and checked 2026-09-25)** — these five links are SUPPORTED:

| Link | Source | What it states |
|---|---|---|
| data-center construction → electricity demand | IEA, *Energy and AI* press release (2025-04-10) | Data-centre electricity demand to more than double by 2030 to ~945 TWh; AI the most significant driver |
| (same link) | U.S. DOE / LBNL 2024 report (2024-12-20) | Data centers 4.4% of U.S. electricity in 2023 → ~6.7–12% by 2028 |
| accelerator demand → HBM demand | SK hynix 4Q24 results (2025-01-23) | HBM >40% of DRAM revenue; HBM demand to keep rising as big-tech AI-server investment grows |
| accelerator demand → foundry utilization | TSMC Q4 2024 earnings call (2025-01-16), **secondary transcript** (The Motley Fool); TSMC's PDF was not machine-readable here | AI-accelerator revenue more than tripled in 2024; gross margin up mainly on higher utilization |
| capacity expansion → equipment demand | SEMI (2025-04-09) | 2024 equipment sales +10% to $117.1B, fueled by leading-edge logic, advanced packaging and HBM capacity |
| AI investment → spender free cash flow | Definitional identity | FCF = operating cash flow − capex |

The other 14 AI links are **PROPOSED**. This includes electricity → grid investment: I found no source quickly that verifies it, so none is cited.

Compact seeds for **crude oil** (with the shale balancing loop, and a low-confidence route through inflation into the policy path) and **monetary policy** (bank earnings MIXED: margin ↑ vs credit losses ↑) mean all three Phase A examples produce a graph. These links are all PROPOSED. Seven Phase A channels are reported as unseeded. The library is validated at import: seeded shifts must match Phase A's channel table exactly, every state must be declared, and every seed must carry its rule id.

## 8. UX

In each News Impact Triage card, below "Why these domains?":

- **A one-line summary.** Example: "Mechanism graph: AI infrastructure investment ↑ → AI compute demand ↑ · Data-center construction ↑ · AI spenders' free cash flow ⇅ … 14 downstream states, 2 with paths that disagree · 5 of 19 links backed by a verified source."
- **An "Economic mechanism graph · N states · M links" expander with four tabs:**
  - **Map**: a top-to-bottom Graphviz layout (rendered client-side; no new Python dependency). Solid = verified source, dashed = reasoned, amber = unresolved/mixed, ↑/↓/⇅ = implied direction. It deliberately uses no verdict green or red.
  - **Consequences**: state, implied direction, slowest link, weakest link, paths, and the chain.
  - **Links & sources**: every link with its sign, status, lag, confidence and backing, then each verified source with the exact statement it makes.
  - **Open questions**: MIXED states with both paths explained, feedback loops, cautions, unseeded channels, and the count of links that are still hypotheses.
- **An unseeded channel** shows one caption and no empty map.

This was verified live with `streamlit run` and Playwright screenshots. That pass changed the map from left-to-right to top-to-bottom with wrapped labels, because six ranks left to right were scaled below legibility. It also reordered the Consequences columns, which had pushed slowest/weakest link off-screen. At phone width (420px) the map is still small; the Consequences tab is the readable view there.

## 9. Tests

- `tests/python/test_news_alpha_phase_b.py` (58 tests) covers:
  - the typed model;
  - verification that has to be earned;
  - no status field on claims;
  - disjointness from the quant vocabulary;
  - both prompt chains, and sign flip on a spending cut;
  - library-driven (not hard-coded) graphs;
  - duplicates across aliases;
  - lag disagreement and state-definition conflicts;
  - undeclared states;
  - direction conflicts (kept, UNRESOLVED, MIXED downstream) and anchor conflicts;
  - missing and observed provenance;
  - model proposals: they stay PROPOSED even when citing a source, can't smuggle a status, origin or verification, are promoted only by an independent verification, join the event graph, and are reported when disconnected;
  - unknown lag/confidence and unsigned links;
  - balancing, reinforcing and indeterminate loops, and self-loops;
  - termination on cycles;
  - JSON round trip and deterministic fingerprints, and a plane that can't be relabelled;
  - the three Phase A examples, a paraphrase, no-cue relative graphs, unseeded and unrelated events, and a cached official item;
  - mandate independence (four mandates, and with the assessments removed);
  - the hop limit and library validation;
  - import boundaries both ways;
  - no forbidden fields;
  - an unchanged registry hash;
  - a map with no verdict colours.
- `tests/python/test_news_alpha_phase_b_ui.py` (3 AppTest tests) covers:
  - the full card;
  - an Equity-only mandate still showing non-tradable states;
  - an unseeded channel.

  Every one of them fails if a live LLM client is constructed.
- Phase A's static guards now parametrize over the new modules automatically (64 → 70).
- **Negative controls:** disabling the conflict rule breaks exactly the two conflict tests, and leaking the mandate into the graph breaks the three mandate-independence tests.

## 10. Files

| New | Purpose |
|---|---|
| `news_alpha/transmission.py` | Typed graph, `assemble_graph`, safety checks, loops, reachability, proposal ingestion |
| `news_alpha/transmission_library.py` | State catalog, seeded claims, verified sources, anchor rules, import-time validation |
| `news_alpha/mechanism_graph.py` | Scan → anchors → reachable subgraph → sign propagation → `EconomicMechanismGraph` |
| `ui/mechanism_graph_view.py` | Summary line, DOT builder, four-tab renderer |
| `tests/python/test_news_alpha_phase_b.py`, `test_news_alpha_phase_b_ui.py` | Tests |

| Changed | Why |
|---|---|
| `news_alpha/__init__.py` | Pipeline docstring and exports |
| `ui/news_alpha_context.py` | `mechanism_graph(scan)` accessor |
| `ui/views/agent.py` | Summary caption and expander in the triage card |
| `scripts/news_alpha_phase_a_demo.py` → `scripts/news_alpha_pipeline_demo.py` | `git mv`; now prints the graph stage and `--scripted-proposal`. The only tracked reference, the Phase A doc, was updated. |

**Deleted: nothing.** No existing helper became obsolete. `translation.mechanism_library` still serves the Phase 1 translation, and `alpha_graph` still serves the Research Map; neither was modified. No C++, registry schema, identity, BH-FDR or `ReliabilityPolicy` change was made. There was no holdout access and no paid call. The source checks were free public web fetches.

## 11. Not done, by design

- **Signal Path Discovery.** Done in Phase C, together with the `MechanismAdjustedImpactAssessment`. Mapping economic states onto tradable markets is Asset Expression (Phase D), not started.
- **A live Claude proposer.** `TransmissionProposal` is the closed schema such an agent would emit, and its ingestion is tested with scripted proposals. Wiring an agent (a new paid-API path) is a separate step. Phase A's static guard also keeps `news_alpha` LLM-free.
- **Persistence.** Graphs are pure and cheap, so they are recomputed per render like the scan. They serialize to JSON if a later stage needs to store them.
- **More sources and seeds.** Citations for the proposed AI links, and seeds for the seven unseeded channels, are data additions to `transmission_library.py`.
