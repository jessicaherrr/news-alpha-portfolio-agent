# News Alpha Portfolio Agent — Phase C: Signal Path Discovery

Status: **IMPLEMENTED, TESTED — stopped at review gate.** Asset Expression (Phase D) not started.

Phase C extends the same pipeline by one stage and adds a second triage pass:

```
ResearchMandate → Allowed Asset Universe → News/Event → Initial Impact Scan → Economic Mechanism Graph
   (Phase A)            (Phase A)            (existing)        (Phase A)               (Phase B)
      → Signal Path Discovery → Direct / Supply-chain / Cross-sector → Economic Consequence   (Phase C, new)
      → Mechanism check of the initial triage, beside it (levels kept)                      (Phase C, new)
```

Run it:

```
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py                   # offline vertical slice
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py --no-cached --scripted-proposal
streamlit run python/alpha_agent/ui/app.py                                    # Agent → News Impact Triage → a card
```

A signal path ends at an **economic consequence** (copper demand ↑) and a **target concept** (copper as an industrial metal). It never names an instrument, a trade, a weight or an expected return. Choosing how to express a consequence in a market is Phase D.

## 1. Reuse audit (the cleanup the prompt asked for)

| Searched for | Found | Decision |
|---|---|---|
| Direct mechanism mapping | `translation.mechanism_library`: `NewsCategory` → **quant** mechanism templates (TREND, CARRY …) for the Phase 1 translation | Different responsibility (quant factor translation). Untouched. |
| `research_angles` | `discovery.research_angles`: *why* a quant hypothesis is generated (baseline replication, failure-guided, cross-market …) | Different responsibility. Untouched. |
| Discovery mechanism helpers | `discovery.mechanisms`: quant-mechanism diversity ordering | Different responsibility. Untouched. |
| Translation helpers | `news_alpha.handoff` → Phase 1 translation | Unchanged; still hands off from the initial scan. |
| Any supply-chain / cross-sector code | None outside `news_alpha` | Gap confirmed; filled here. |
| Existing path enumeration | Phase B's `StateImplication.paths` (`mechanism_graph._enumerate_paths`) | **Reused** as the extraction. The inline construction became the public `mechanism_graph.trace_path`, so validated model paths are built by the same code. Behaviour and fingerprints are identical. |
| Phase A's per-exposure scoring | `impact_scan._level_rank` | Exposed as the public `impact_scan.impact_level_for` (same rule). Since the acceptance patch the second pass changes no level, so only the initial scan uses it; it remains the single entry point for any future magnitude-based revision. |

**Deleted: nothing.** No helper became obsolete, and nothing had zero references. `alpha_graph` (the evidence graph) is untouched and still test-asserted separate from `news_alpha`.

## 2. The typed `SignalPath` (`news_alpha/signal_paths.py`)

| Prompt field | Field(s) |
|---|---|
| path_id | `path_id` is event-scoped. `signature` is structural and identical for the same route from any event. |
| origin_event_id / mechanism_graph_id | `origin_event_id`, `mechanism_graph_id` (the graph's fingerprint) |
| ordered nodes/edges | `state_ids`, `state_labels`, `links` (`PathLink`: sign, channels, lag, confidence, support, origins, value-chain role). A validator enforces that the links chain the states in order. |
| path_type | `path_type`: the primary label (`None` = unclassifiable, never guessed). `classification_basis` gives it in words. The two characteristics behind it are recorded separately: `value_chain_steps` and `sector_transitions` (see §3). `sectors` lists the sector of each state. |
| economic consequence | `consequence_state`, `consequence_label`, `consequence_kind` |
| candidate target concept | `candidate_target_concept` |
| expected direction | `expected_direction` (the consequence's movement), `relative_sign` (known even without an event direction) |
| expected horizon | `expected_horizon`: the **economic** transmission horizon (slowest link). This is not a market-reaction horizon. |
| transmission_depth | `transmission_depth` (= number of links) |
| confidence | `confidence` (weakest link), `weakest_support` |
| provenance | `provenance`: origin, rule ids, seed rules, verified sources, model outputs |
| status | `status` + typed `status_reasons`, `opposing_path_ids` |

`SignalPathDiscovery` holds every path for one event, plus conflicts, rejected model proposals and findings. Its `plane` is fixed to `"HYPOTHESIS"`.

## 3. Path types (`path-type/1`)

The type is computed from reviewed data (`news_alpha/signal_path_library.py`), never from keywords:

- **`STATE_PROFILES`** give every catalog state an `EconomicSector` and a target concept. Semiconductors, hyperscalers and data-center construction form one technology complex; electricity is utilities; copper is materials.

  **`EconomicSector` is an internal transmission taxonomy.** It exists only to answer one question: does an effect leave the economic sphere where it first landed? It is **not** GICS, NAICS, SIC or any other industry standard. It is not the Equity universe's sector groups, and it is **not an Asset Expression mapping**: no sector selects, weights or names a tradable instrument. Its boundaries are transmission judgments. For example, "monetary and rates" is a sphere that no classification standard has. Changing it is a reviewed data edit, checked by the path-type tests.
- **`CHANNEL_ROLES`** mark whether a transmission channel moves the effect to a new counterparty: SUPPLIER (`INPUT_DEMAND`, `CAPACITY_UTILIZATION`) or CUSTOMER (`COST_PASS_THROUGH`). `DEMAND_PULL` has no role, because it is the event's own spending re-expressed as demand.

Rule, applied in order:

1. **CROSS_SECTOR**: a later state lies in a different sector from the path's **first consequence**. The reference is where the effect first lands, not the anchor, so a rate hike → bank margins → bank earnings stays DIRECT.
2. **SUPPLY_CHAIN**: otherwise, the path takes ≥ 2 supplier/customer steps. The effect has gone past the event's first counterparty.
3. **DIRECT**: otherwise. These are the event's own first-order targets and its first supplier or customer.

| Event | DIRECT | SUPPLY_CHAIN | CROSS_SECTOR |
|---|---|---|---|
| AI capex ↑ | compute, accelerators, data-center construction, spenders' FCF, services revenue (6) | HBM, foundry utilization, capacity expansion, semi equipment (7) | electricity, grid, power equipment, gas for power, copper (7) |
| OPEC+ cut | crude price, refined products, producers' revenue, shale drilling (4) | none (said so in the UI) | inflation, transport fuel costs, the whole rates branch (13) |
| Fed hike | yields, USD, borrowing costs, bank margins, bank earnings (5) | none | equity multiples, gold, housing, credit losses, bank earnings via credit (5) |

**`PathType` is a primary label over two independent characteristics.** Precedence decides the label (CROSS_SECTOR, then SUPPLY_CHAIN, then DIRECT). Both characteristics are recorded on every path, so the label never erases either one:

- `value_chain_steps`: supplier/customer steps along the whole path, in any sector;
- `sector_transitions`: sector changes after the first consequence (`None` if a state is unclassified).

AI capex → data centers → electricity → gas for power is labelled CROSS_SECTOR but carries 2 value-chain steps and 2 transitions; its basis says so in words. A SUPPLY_CHAIN path always has 0 transitions. The table shows both columns, and a validator enforces that CROSS_SECTOR holds exactly when there is a transition. A test shows all four combinations occur in the AI graph.

Both prompt examples come out as specified. AI compute demand → accelerator demand is DIRECT (one supplier step). → HBM → foundry → equipment is SUPPLY_CHAIN. Data centers → electricity → grid equipment is CROSS_SECTOR.

## 4. Depth is a research variable, not a score

`transmission_depth` and `value_chain_steps` are recorded on every path. **Neither enters status, impact level or ordering** beyond reading order (type, then depth). A test pins this down: a one-link path (AI services revenue) is PROPOSED while a five-link path (semiconductor equipment) is RESEARCHABLE, and no path of any depth moves an impact level.

Whether information diffuses differently by path type and horizon (1D/5D/20D/60D) is recorded for later study, not answered.

## 5. Status (`path-status/1`)

| Status | When (typed `StatusReason`) |
|---|---|
| **UNRESOLVED** | `UNRESOLVED_LINK`, `UNSIGNED_LINK`, `ANCHOR_CONFLICT`, `UNCLASSIFIED_STATE`, `OPPOSED_AT_SAME_HORIZON` |
| **PROPOSED** | `NO_EVENT_DIRECTION`, `WEAK_LINK` (low/unknown confidence), `UNKNOWN_LAG`, `UNREVIEWED_MODEL_LINK` |
| **RESEARCHABLE** | none of the above: a direction, a horizon, and reviewed links of at least medium confidence |
| **REJECTED** | only a model path proposal that failed validation (`RejectedPathProposal`, preserved) |

RESEARCHABLE means ready to take further, **not shown to be true**. The UI shows "Verified links: 1 of 3" beside it. A validator refuses any status that does not follow from its reasons.

**Conflicting paths** (paths reaching one consequence with opposite directions) are all kept and cross-referenced, never averaged:

- **Different horizons → separable.** Spenders' FCF falls within months (capex) and rises over years (services revenue). Both stay alive, because a horizon tells them apart.
- **Same horizon → both UNRESOLVED.** Bank earnings rise via margins and fall via credit losses, both over quarters.
- **Only a *sound* opposing path counts.** A sound path has no unresolved, unsigned or unreviewed-model link. The live demo exposed the need for this: without the rule, one model-proposed contradiction on electricity → grid unresolved the independent reviewed path *data centers → power equipment*. It is still listed as opposition.

## 6. Duplicates and model path proposals

- **Within an event:** the route's signature identifies it, so one route is one path.
- **Across events:** a paraphrased event yields the same signatures with different `path_id`s. The same hypothesis is recognisable, the event provenance is separate.
- **Model paths:** `SignalPathProposal` / `ProposedSignalPath` is the only shape an LLM's path output may take. It carries state names and a rationale, and nothing else (`extra="forbid"`: no type, status, direction or instrument field). Each proposal ends in one of three ways:
  - a **duplicate** of an extracted path: recorded as provenance (`DUPLICATE_PROPOSAL`), never added, status unchanged;
  - a **valid new walk** over existing links: added as `MODEL_PROPOSED` (`MODEL_PATH_ADDED`);
  - **REJECTED** with typed reasons: `UNKNOWN_STATE` (e.g. "NVDA", "Copper futures"), `NOT_FROM_AN_ANCHOR`, `NOT_A_GRAPH_LINK`, `REPEATS_A_STATE`, `TRADE_EXPRESSION`.
- **A model cannot add a link or a state through a path.** Links arrive only as Phase B `TransmissionProposal` claims, with their own provenance. A path through such a link is `UNREVIEWED_MODEL_LINK`, and unclassified if its state is new.
- **The trade lexicon is a secondary guard.** The real guard is structural: a proposal can only name graph states. "Sell AI services" in an economic rationale is not flagged (tested).

No LLM is wired to this schema. `news_alpha` stays LLM-free, and scripted proposals exercise it.

## 7. Mechanism-adjusted impact (`news_alpha/impact_adjustment.py`, `mechanism-adjusted-impact/2`)

A separate `MechanismAdjustedImpact` revisits every Phase A exposure. The initial `InitialImpactScan` is **never modified**: the second pass keeps its fingerprint and every initial level beside the adjusted one.

**Two quantities, kept apart.**

- **Impact relevance:** how much of an exposure's economics the event moves.
- **Mechanism support:** whether a reviewed route from the event reaches the exposure's economic basis, and how confident its weakest link is.

A high-confidence link says the relationship probably holds, not that its effect is large. So link confidence is **never** mapped onto exposure strength and never moves an impact level.

> Acceptance fix. Rule /1, in the first Phase C commit, mapped HIGH/MEDIUM/LOW link confidence onto PRIMARY/SECONDARY/TERTIARY strength and moved levels by one step. It raised AI copper LOW → MEDIUM and lowered Fed equity index futures VERY HIGH → HIGH. That conflated causal confidence with economic exposure, and is withdrawn.

Rule /2, per exposure:

1. **Basis.** `EXPOSURE_BASIS` records the economic states an exposure's own rationale rests on, e.g. copper futures → copper demand. Every exposure of a seeded channel must have one (checked at import). Only paths from the **same channel's** anchor count.
2. **Mechanism support** (`MechanismSupport`) is reported, never scored:
   - `CORROBORATED`: a researchable path reaches the basis;
   - `PROPOSED_ONLY`: paths reach it, but none is researchable yet;
   - `NOT_REACHED`: a missing link is a library gap, not evidence against;
   - `CHANNEL_NOT_SEEDED`;
   - `NO_ECONOMIC_BASIS`.

   The cited route is chosen as researchable first, then unopposed, then most confident, and never by depth. Its weakest-link confidence and backing are carried as `mechanism_confidence` / `mechanism_backing`: properties of the route, not of the exposure.
3. **Level.** Nothing in the graph establishes **economic magnitude** (exposure share, elasticity, size of effect). So `adjusted_level == initial_level`, `magnitude_established = False`, and every revision carries that uncertainty (`MAGNITUDE_NOTE`). A future pass may move a level only from an established magnitude input, and then by at most one step. `LevelChange.RAISED` / `LOWERED` stay in the vocabulary for that and are unreachable today.

| Event | Levels | Mechanism check |
|---|---|---|
| AI capex ↑ | all kept (Futures MEDIUM, ETF MEDIUM, Equity HIGH) | 7 of 7 exposures corroborated. Utilities ETF: an all-high-confidence route, yet its level stays LOW, because a sure route is not a large effect. |
| OPEC+ cut | all kept | crude/refined futures, USO, energy and industrials all corroborated |
| Fed hike | all kept | Treasuries, equity index, FX, credit, financials and growth corroborated. Gold is proposed-route-only (its yields → gold link is low confidence). |
| Geopolitical (unseeded) | all kept | channel not seeded |

This is the smallest conservative rule. It never contradicts the initial scan, and it states what the graph adds (a route, and how sure we are of it) next to what it cannot add (a magnitude).

To make this possible, Phase A's `InitialImpactAssessment` gained an additive `contributions` field (`ExposureContribution`: channel, exposure, strength, detection, salience, level, symbols), so the second pass checks exposure by exposure instead of re-deriving from the flattened candidate list.

## 8. UX (Agent → News Impact Triage card)

- **Two rows:** the *Initial scan* badges, then *After mechanism graph* in plain text: "levels kept (economic magnitude not established) · mechanism corroborates exposures: Futures 3 of 3 · ETF 2 of 2 · Equity 2 of 2". There is no impact badge or arrow on the second row, because it changes no level.
- **A signal-path caption:** "Signal paths: 6 direct · 7 supply-chain · 7 cross-sector — 17 researchable." It also names any consequence with opposing paths at the same horizon.
- **A "Signal paths · N hypotheses · M researchable" expander** with Direct / Supply chain / Cross-sector tabs (plus Unclassified and Rejected proposals only when present). Each tab has a one-line definition and a table:
  - columns: consequence · direction · status · depth · value-chain steps · sector crossings · horizon · weakest link · verified links · via · target concept;
  - a non-researchable status carries its reason in the same cell;
  - a consequence reached by several routes gets a route hint ("· via HBM demand") so rows are never visually identical;
  - opposing-path notes appear under the table;
  - an empty type says why ("No supply-chain path …").
- **"Why these domains?"** gains a *Mechanism check (second pass)* table. Level and mechanism evidence sit in separate columns: exposure, initial level, after graph, mechanism, route confidence, economic basis with ↑/↓/⇅.
- **The Agent chat** reply for a described event adds the signal-path line and the second-pass levels, with both objects in `evidence`.

This was verified live with `streamlit run` and Playwright at 1440px and 420px. That pass led to three changes:

- the reason column moved into the Status cell, because it had been scrolled off-screen;
- route hints were added, because two "Semiconductor equipment demand" rows looked identical;
- the cited route now prefers an unopposed path, because the Nasdaq rationale had cited the contested FCF path.

At 420px the tables scroll horizontally, as Phase B's do.

## 9. Tests

- **`tests/python/test_news_alpha_phase_c.py` (58 tests after the acceptance patch):**
  - classification of all three types, from where the effect first lands;
  - the oil case with no supply-chain path;
  - classification driven by the tables (re-sectoring copper and removing a role change the types);
  - unclassified states;
  - one ordered signal path per graph path;
  - validator refusals for shuffled links, wrong depth and inconsistent status;
  - JSON round trip and fingerprint determinism;
  - depth never ranked;
  - mandate independence;
  - unseeded events;
  - the typed status reasons, including no event direction and anchor conflict;
  - both conflict regimes, and a disputed link that does not unresolve independent paths;
  - duplicate model proposals, and signatures shared across events with distinct path ids;
  - a valid model walk being added;
  - seven parametrized rejections, plus "sell AI services" not being flagged;
  - smuggled fields refused;
  - provenance from the links, and model provenance;
  - consequence fields;
  - the initial scan never modified, with both levels kept;
  - link confidence never becomes exposure strength or moves a level (including an all-high-confidence route to a tertiary exposure);
  - mechanism support reported beside the level (corroborated, proposed-only, not reached, not seeded), own-channel paths only;
  - `PathType` as a primary label over independent `value_chain_steps` / `sector_transitions`;
  - the sector table documented as an internal transmission taxonomy;
  - excluded domains;
  - library validation;
  - target concepts named without any real universe symbol;
  - import boundaries, no trade/instrument/verdict fields, and an unchanged registry hash;
  - the view helpers.
- **`tests/python/test_news_alpha_phase_c_ui.py` (3 AppTest/chat tests):** the full card, an unseeded card, and the chat evidence. Every test fails if a live LLM client is constructed.
- **Phase A's static guards** automatically cover the new modules.
- **Negative controls (each broke exactly its own test, then was restored):**
  - counting unsound opposition broke the disputed-link test;
  - dropping same-horizon opposition broke the bank-earnings test;
  - counting `DEMAND_PULL` as a value-chain step broke the three classification tests.

## 10. Files

| New | Purpose |
|---|---|
| `news_alpha/signal_paths.py` | `SignalPath` types, discovery, status, conflicts, dedupe, proposal validation |
| `news_alpha/signal_path_library.py` | Reviewed sectors, target concepts, channel roles, exposure bases; import-time validation |
| `news_alpha/impact_adjustment.py` | `MechanismAdjustedImpact` second pass |
| `ui/signal_path_view.py` | Badge rows, signal-path tabs, second-pass table, chat text |
| `tests/python/test_news_alpha_phase_c.py`, `test_news_alpha_phase_c_ui.py` | Tests |

| Changed | Why |
|---|---|
| `news_alpha/schemas.py`, `impact_scan.py` | Additive `ExposureContribution` record; public `impact_level_for` (same rule) |
| `news_alpha/mechanism_graph.py` | Public `trace_path` (same construction; fingerprints unchanged) |
| `news_alpha/__init__.py` | Pipeline docstring, exports |
| `ui/news_alpha_context.py`, `ui/views/agent.py`, `ui/conversation_engine.py` | Accessors, card, chat |
| `scripts/news_alpha_pipeline_demo.py` | Signal-path and second-pass stages; scripted path-proposal fixture |

There was no C++, registry schema, identity, BH-FDR or `ReliabilityPolicy` change, no holdout access, and no paid or network call.

## 11. Not done, and choices to review

- **Asset Expression (Phase D).** Consequence → instrument, direction on a price, options structure: not started, by design.
- **Settled at acceptance:**
  1. `SUPPLY_CHAIN_MIN_STEPS = 2`: the first counterparty is direct.
  2. The sound-opposition rule is kept.
  3. The sector table is an internal transmission taxonomy.
  4. Link confidence never sets relevance.
- **Economic magnitude.** Levels can only move once a magnitude input exists: exposure share, revenue mix, elasticity. None is modelled. When one is, the one-step bound applies.
- **Discovered exposures.** The oil graph reaches the rates branch, but the second pass only revises exposures the initial scan assessed. It does not add rate exposures to an oil event. That would need the mandate universe in the second pass and a prior for unseen exposures.
- **A live Claude path/link proposer.** The closed schemas and their validation are ready; wiring one is a paid-API step.
