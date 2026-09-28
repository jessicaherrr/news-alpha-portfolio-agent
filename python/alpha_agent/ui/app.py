"""Phase 20.1 -- Streamlit research interface entrypoint.

A presentation/control layer over the existing deterministic system (CLAUDE.md
"Architecture boundaries"). It may submit research objectives and invoke
approved service/orchestrator interfaces (the real Phase 16/17 agents, the
Phase 14 registry's read API); it never owns or recomputes PnL, fills, risk,
validation, scientific verdicts, experiment identity, or holdout decisions, and
it exposes no control that bypasses frozen validation, risk, duplicate
authority, multiple-testing control, or 2025 holdout isolation.

Launch with:

    streamlit run python/alpha_agent/ui/app.py

(run from the repository root, with the `ui` extra installed:
``pip install -e '.[ui]'``). See docs/STREAMLIT_RESEARCH_INTERFACE.md.

RUNTIME CONNECTIVITY (HOME TERMINAL pass, mission section 1): a normal
`streamlit run` of THIS file discovers a local `.env` (``DATABENTO_API_KEY``,
``ANTHROPIC_API_KEY``, ...) automatically -- see
`bootstrap_local_runtime_environment` below. `.env` stays untracked
(`.gitignore`); nothing here ever reads its content into a string this
process prints, logs, persists to the registry, or displays in the UI --
`python-dotenv` writes straight into `os.environ`.
"""
from __future__ import annotations


def bootstrap_local_runtime_environment() -> None:
    """The ONE canonical place a local `.env` is loaded for this product's
    runtime. Called explicitly, once, at real app startup (`main`, below) --
    deliberately NOT at arbitrary module import time. That distinction is
    the actual fix for a real bug: a local `DATABENTO_API_KEY` used to reach
    this app only via a manual `source .env` before `streamlit run`, because
    nothing in the normal startup path ever loaded it, while a handful of
    test modules called the same underlying `load_dotenv_if_available()` at
    IMPORT time -- which (only when a real key happened to be present)
    leaked it into the whole pytest process and turned ordinary,
    should-be-offline UI tests into real network calls. The fix is call-SITE
    discipline, not a new mechanism: reuse the same safe, idempotent,
    never-raising loader, but only from this one explicit, real-runtime
    entrypoint -- see `tests/python/conftest.py` for the independent,
    defense-in-depth guarantee that ordinary tests stay offline regardless
    of what is or is not in `os.environ` when they run."""
    from alpha_agent.knowledge.connector_support import load_dotenv_if_available

    load_dotenv_if_available()


def main() -> None:
    bootstrap_local_runtime_environment()

    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install UI extras: pip install -e '.[ui]'") from exc

    from alpha_agent.ui.views import (
        agent,
        backtests,
        crypto_lab,
        dashboard,
        discover,
        experiment_log,
        learn,
        market,
        news,
        paper_trading,
        portfolio,
        research,
        strategies,
        system,
        validation,
        workspace,
    )

    st.set_page_config(
        page_title="Agentic Alpha",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Text-only navigation, deliberately: no page carries a decorative emoji
    # icon. The sidebar is a fully custom nav (`layout.render_sidebar_nav`)
    # with monochrome inline-SVG icons and progressive-disclosure sub-groups;
    # `position="hidden"` keeps `st.navigation`'s real multipage ROUTING (URL
    # paths, `st.switch_page`, page identity) while rendering none of its own
    # list -- every page calls `layout.render_sidebar_nav(active=...)` first.
    #
    # News Alpha Research Thread workspace: the research journey is the
    # primary navigation -- News (the default landing: "what is worth
    # researching?"), Research (the persistent Research Thread workspace),
    # Market (ONE market destination: the active thread's Affected Markets and
    # Explore All Markets), Portfolio (across threads) and Learn (My Alpha /
    # Research Map / Community / Guides). Ask (the former Agent landing),
    # Strategy Lab (the former Research page, now at `lab`) and Paper Trading
    # are secondary Tools. Every former hidden page stays registered hidden and
    # reachable by its existing in-app hand-offs.
    pages = {
        "Agentic Alpha": [
            st.Page(news.render, title="News", url_path="news", default=True),
            st.Page(workspace.render, title="Research", url_path="research"),
            st.Page(market.render, title="Market", url_path="market"),
            st.Page(portfolio.render, title="Portfolio", url_path="portfolio"),
            st.Page(learn.render, title="Learn", url_path="learn"),
            st.Page(agent.render, title="Ask", url_path="agent"),
            st.Page(research.render, title="Strategy Lab", url_path="lab"),
            st.Page(paper_trading.render, title="Paper Trading", url_path="paper-trading"),
            st.Page(system.render, title="Settings", url_path="system"),
        ],
        "": [
            st.Page(discover.render, title="Discover", url_path="discover", visibility="hidden"),
            st.Page(dashboard.render, title="Dashboard", url_path="dashboard", visibility="hidden"),
            st.Page(strategies.render, title="Strategies", url_path="strategies", visibility="hidden"),
            st.Page(backtests.render, title="Backtests", url_path="backtests", visibility="hidden"),
            st.Page(validation.render, title="Validation", url_path="validation", visibility="hidden"),
            st.Page(experiment_log.render, title="Experiment Log", url_path="experiment-log", visibility="hidden"),
            st.Page(crypto_lab.render, title="Crypto Lab", url_path="crypto-lab", visibility="hidden"),
        ],
    }
    nav = st.navigation(pages, position="hidden")
    nav.run()


if __name__ == "__main__":
    main()
