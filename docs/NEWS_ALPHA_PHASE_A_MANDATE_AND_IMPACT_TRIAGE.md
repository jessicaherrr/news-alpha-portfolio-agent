# News Alpha Portfolio Agent — Phase A: Research Mandate + Initial Impact Scan

Status: **ACCEPTED** (commit `eb2d8c3`). The next stage, the Economic Mechanism Graph, is
Phase B: see `docs/NEWS_ALPHA_PHASE_B_ECONOMIC_MECHANISM_GRAPH.md`.

Phase A builds the front of the canonical pipeline and connects it to the
existing Phase 1 translation layer:

```
ResearchMandate → Allowed Asset Universe → News/Event → Initial Impact Scan → existing translation boundary
   (new)              (new scope)          (existing)        (new, pass 1)      (existing, unchanged)
```

This implements sub-phase 10.1 of the untracked proposal
`docs/PHASE_10_SIGNAL_PATH_AND_PORTFOLIO_CONSTRUCTION_PROPOSAL.md` (mandate + cross-domain
universe), plus the Initial Impact Scan that the proposal had mapped to Phase 1.

Run it:

```
PYTHONPATH=python python scripts/news_alpha_pipeline_demo.py    # offline vertical slice (renamed from news_alpha_phase_a_demo.py in Phase B)
streamlit run python/alpha_agent/ui/app.py                       # Agent → News Impact Triage
```

## 1. Where each stage lives

| Stage | Module | New / reused |
|---|---|---|
| Research Mandate | `alpha_agent.news_alpha.mandate` | New. Composes `recommendation.profile.InvestorProfile` verbatim. |
| Allowed Asset Universe | `alpha_agent.news_alpha.universe` | New scope over `marketdata.product_catalog`, `etf.universe`, `equities.universe`. |
| News / Events | `alpha_agent.market_intel` | Reused. `news_alpha.events` is a read-only view; no second store. |
| Initial Impact Scan | `alpha_agent.news_alpha.channels` + `.impact_scan` + `.schemas` | New. |
| Translation hand-off | `alpha_agent.news_alpha.handoff` → `translation.pipeline` | New adapter into the unchanged Phase 1 entry point. |
| UI | `ui/news_alpha_context.py`, `ui/views/agent.py` | New boundary module; Agent's Research Translation section now opens with triage. |

## 2. Decisions a reviewer should check

**Mandate reuses the profile rather than duplicating it.** Horizon, risk
style, drawdown, turnover and capital already live on `InvestorProfile`, which
has its own editor and store. `ResearchMandate.risk_profile` *is* that object.
`MandateStore` persists only the access/constraint half
(`data/user_prefs/research_mandate.json`, gitignored). The risk half is always
the current profile, so there are never two copies that could disagree.

**Access and risk are separate axes, and tests enforce it.** `allowed_domains`
and the instrument allow/deny lists decide *scope*. Risk style, drawdown,
leverage, shorting and horizon never change an impact level, availability,
candidate list or direction; they only add `mandate_notes`, such as "downward
pressure cannot be expressed as a net short". No code maps a domain to a risk
level or a risk level to a domain.

**`MandateDomain`, not a widened `AssetDomain`.** `registry.enums.AssetDomain`
(FUTURES/ETF/EQUITY) is the registry's evidence namespace, and widening it is
a FROZEN RESEARCH SEMANTICS stop. A user still has to be able to exclude
Options and Crypto, so `MandateDomain` is the access vocabulary. It has every
`AssetDomain` member under the identical value string (test-enforced), plus
OPTIONS and CRYPTO, which map to `None`.

**Domain support is platform truth, reported honestly:**

| Domain | Support today |
|---|---|
| Futures | RESEARCH_READY — 5 certified roots; the other 31 catalogued roots are observation-only |
| ETF | RESEARCH_READY — Phase 6 pilot data present (probed from the raw store; DATA_NOT_ACQUIRED otherwise) |
| Equity | DATA_NOT_ACQUIRED — Phase 9.1 universe declared, the ~$0.44 acquisition not yet approved |
| Crypto | SYNTHETIC_ONLY — CME crypto futures (BTC/MBT/ETH) remain reachable via Futures |
| Options | NOT_SUPPORTED — not assessed |

**The impact-level rule (`impact-level/1`)** applies to each detected channel
× exposure that has at least one candidate market in the scoped universe:

- Base level from exposure strength: PRIMARY → HIGH, SECONDARY → MEDIUM, TERTIARY → LOW.
- One level lower when detection is WEAK (a single cue).
- One level higher when detection is at least MODERATE and the event is a
  *surprise*. That means a surprise term, or a scheduled event that
  market_intel's importance rule marks HIGH. Superlatives such as "record" or
  "largest" do not count: real EIA explainers use them.
- The domain level is the maximum over its contributions.

Detection strength works like this. A source category from market_intel's own
deterministic mapping is STRONG. Two or more lexicon/issuer cues are MODERATE.
A single cue is WEAK. Confidence is categorical (LOW/MEDIUM/HIGH) and means
confidence in the *classification*, never in any outcome. There is no numeric
score.

**Direction only when justified.** A price sign appears only when (a) exactly
one shift's cue phrases matched and (b) the channel table records a textbook
first-order sign for that market group, for example Treasury futures prices
DOWN on a tightening surprise. AI capex carries no sign, because spenders and
suppliers sit in the same sectors; the Mechanism Graph resolves that.
Conflicting cues leave the direction undetermined.

**The hand-off has three fidelity levels (`CategoryBasis`):**

- **SOURCE_CATEGORY:** a real item's own category already covers the root, so
  `observation_from_news` / `observation_from_event` is used verbatim, with
  impact provenance appended to `evidence_refs`.
- **IMPACT_CHANNEL:** there is no source category (a user-described OPEC cut,
  or a Fed speech filed as OTHER). The channel routes the item to the Phase 1
  templates for the same channel, with an explicit "approximation" fidelity note.
- **NO_TEMPLATE:** no deterministic template exists yet, so the translation
  reports its own honest research gap. The Claude opt-in stays available.

Only certified Futures roots can be handed off, because the Phase 1
`Observation` validator accepts nothing else. Every other relevant domain gets
a typed `TranslationGap` instead of being dropped silently.

## 3. Vertical slice (mandate: Equity + Futures)

| Event (user-described) | Futures | Equity | ETF / Options / Crypto | Hand-off |
|---|---|---|---|---|
| A. AI infrastructure spending increase | MEDIUM (NQ, HG, NG) | HIGH (Tech/Comm Svcs) | excluded | NQ via NO_TEMPLATE → honest research gap; Equity gap |
| B. OPEC+ production cut | HIGH (CL…), UP | MEDIUM (XOM, CVX), UP | excluded | CL via IMPACT_CHANNEL → PETROLEUM → AVAILABLE trend hypothesis |
| C. Fed surprise 50 bp hike | VERY_HIGH (ZN, ES, NQ…), Treasuries DOWN | HIGH (Financials…) | excluded (ETF also VERY_HIGH when allowed) | ZN/ES/NQ/GC via IMPACT_CHANNEL → FOMC templates |

None of these is special-cased. The tests also run paraphrases through the
same triage and assert that the example sentences don't appear in any source file.

## 4. UX

The Agent page's Research Translation section now opens as **News Impact
Triage**:

- A one-line mandate summary, marked "default — not yet personalized" until
  the user saves a mandate.
- An "Edit research mandate" form covering domains, optional per-domain
  instrument limits, shorting, leverage and liquidity. Risk comes from Your Profile.
- A "Describe a market event" box. These events are session-only and never
  written to NewsStore.
- One card per event, most relevant first. Each card shows per-domain badges,
  detected channels, a "Why these domains?" breakdown, and a
  `Translate <ROOT>` button per certified root. The button leads into the
  existing translation card, which is unchanged.

Button keys keep the Phase 1 `agent-translate-go-news:<id>:<root>` format.

It was verified live with `streamlit run` and Playwright screenshots. That
pass caught and fixed one real contradiction: a signed Energy group next to an
unsigned tertiary group had also produced a "no sign applies" note.

## 5. Tests

- `tests/python/test_news_alpha_phase_a.py` (58 tests) covers:
  - mandate serialization, canonicalization, validation and the store;
  - `AssetDomain` coverage;
  - deterministic universe resolution;
  - instrument constraints;
  - channel-table integrity against the live universes;
  - the three examples plus paraphrases;
  - salience, weak cues and conflicting cues;
  - crypto and options handling;
  - risk-vs-access independence, both ways;
  - provenance;
  - the three hand-off bases;
  - an end-to-end translation with the registry file hash unchanged;
  - static guards: no scientific-plane or knowledge imports, no registry or
    NewsStore writes, no network, market_intel never imports news_alpha, no
    return/probability/weight/verdict/alpha field.
- `tests/python/test_news_alpha_phase_a_ui.py` (5 AppTest tests) covers the
  empty state, cached-news triage, describe → scan → translate with no LLM,
  a saved Futures-excluding mandate, and the mandate form saving access fields only.
- `tests/python/conftest.py` gains an autouse fixture that points the mandate
  store at a per-test temp file. No test depends on, or overwrites, a
  developer's real mandate.

## 6. Additive edits to existing modules

- `etf/universe.py`: `EXPOSURE` labels (documentation metadata, mirrors `equities.universe.SECTOR`).
- `etf/data_source.py`, `equities/data_source.py`: `primary_listing_data_acquired()`,
  a cheap path-existence probe with no decode and no network.
- `ui/palette.py`: `IMPACT_*` badge states, a blue ramp. They are prefixed so
  they don't collide with any bare HIGH/LOW, and they deliberately avoid the
  verdict green/red.

No C++ changes, no registry schema, identity, BH-FDR or `ReliabilityPolicy`
changes, no holdout access, and no paid calls.

## 7. Cleanup audit

Nothing was deleted. The old per-(item, root) candidate list inside the
Agent's Research Translation section was replaced in place, and its function
was removed from `views/agent.py`. `translation_context.candidate_observations`
stays: the conversation engine's `OBSERVATION_TO_FACTOR` handler still uses
it for cached-news requests ("the latest FOMC news"); chat-DESCRIBED events
now go through triage instead (see §9).
`docs/PHASE_10_…_PROPOSAL.md` is left untracked, as found.

## 8. Not done, by design

- No Mechanism Graph, no `MechanismAdjustedImpactAssessment`, and no
  supply-chain or cross-sector propagation.
- (Done after Phase A, see §9: chat-described events now route through triage.)
- No Claude-proposed channels. The scan is deterministic only.
- Liquidity and leverage are carried as constraints but not applied. No
  universe has typed liquidity data, and leverage belongs to Portfolio
  Construction.

## 9. Chat integration + cache hygiene (post-Phase-A fix)

**Chat-described events use the Phase A path.** Before this fix, an Agent chat
message such as "OPEC+ agreed a production cut. What could I research?" was
answered by translating whichever cached news item came first. Now
`conversation_engine._handle_observation_to_factor` first asks whether the text
describes an event. It does when two things hold: the text doesn't point at
cached news (no "news", "headline(s)", "release(s)", "cached", "from/about
this"), and the deterministic Phase A scan detects at least one channel in it.
If so:

- The text, minus a trailing research question, becomes a session
  `UserDescribedEvent` via `news_alpha_context.add_user_event`, the same call
  the "Describe a market event" box makes.
- It is scanned once with `news_alpha_context.scan_user_event`, which is
  `scan_initial_impact` under the mandate-scoped universe.
- That same scan feeds the Phase B mechanism graph, the translation hand-off,
  and the scripted translation.
- The event also gets its own triage card on the page.

Otherwise the unchanged Phase 1 cached-item path answers. Event identity is
stable: re-describing the same text, whether in chat or in the box, returns
the same event (same `event_id`). No impact, registry, or translation
semantics changed. Tests: `tests/python/test_news_alpha_chat_event_continuity.py`.

**The leaked `N1` fixture.** A local cache row (`news_id="N1"`,
`https://eia.gov/x`) came from an earlier, pre-isolation version of
`test_translation_context.py`, which is already isolated today. There are two
fixes:

- `tests/python/conftest.py` gains an autouse fixture that points
  `market_intel_context._store` / `_event_store` at per-test temp stores, so
  no test can read or write the real cache again.
- The single fixture row was deleted from the local, gitignored observational
  cache, after taking a backup copy.

