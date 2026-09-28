"""Release UI polish -- the persistent application shell: injected CSS, the
custom top header bar, and the sidebar (brand, page nav, Markets, Data
Status, Research Window). `.streamlit/config.toml` sets the native dark
theme (so `st.dataframe`'s canvas grid, default chart colors, and widget
borders are correctly dark without CSS); this module's CSS only reaches what
that config cannot: sidebar width/nav look, card geometry, typography scale,
badges, and density. Every selector below targets a stable Streamlit
`data-testid` / ARIA attribute or an explicit `st.container(..., key=...)`
class -- never an auto-generated `st-emotion-cache-*` hash.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote as _urlquote

import streamlit as st

from alpha_agent.ui import palette, services

#: Shared main-content : right-rail column ratio (~320px rail on a
#: 1440-1920px desktop canvas once the sidebar and padding are subtracted).
#: Every page that has an AI-agent rail uses this same ratio so the shell
#: feels identical everywhere.
MAIN_RAIL_RATIO = [3.3, 1]

_ASSET_DIR = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def _brand_mark_svg() -> str:
    """The original Agentic Alpha vector mark, inlined from its repo asset
    file (`assets/agentic_alpha_mark.svg`) so it renders crisply at any size
    with no extra HTTP round-trip -- Streamlit does not serve arbitrary
    static files without extra configuration."""
    return (_ASSET_DIR / "agentic_alpha_mark.svg").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sidebar IA pass (Release UX product-consolidation acceptance pass, section
# 6): monochrome, stroke-based, Lucide-inspired line icons -- fill="none",
# stroke="currentColor", stroke-width ~1.7, round linecap/linejoin, 18-20px --
# for the shared sidebar's five primary destinations. No emoji, no Unicode
# pseudo-icons, no icon-font file: each icon is hand-authored inline SVG path
# data, embedded as a CSS `mask-image` data URI keyed to its own row's stable
# `st.container(key=...)` class (never a fragile `nth-of-type` position
# selector) so a real, clickable `st.button` can carry it as a `::before`
# pseudo-element that tracks `currentColor` through hover/selected state --
# Streamlit's own button label cannot embed raw HTML/SVG, so this is the one
# place raw SVG bytes need to leave Python at all (see `render_sidebar_nav`).
# ---------------------------------------------------------------------------

_ICON_PATHS: dict[str, str] = {
    "news": (
        '<rect x="3" y="4" width="14" height="16" rx="2"/>'
        '<path d="M17 8h2.5a1.5 1.5 0 0 1 1.5 1.5V18a2 2 0 0 1-2 2H6"/>'
        '<line x1="6.5" y1="8.5" x2="13.5" y2="8.5"/><line x1="6.5" y1="12" x2="13.5" y2="12"/>'
        '<line x1="6.5" y1="15.5" x2="11" y2="15.5"/>'
    ),
    "research": (
        '<circle cx="5" cy="5.5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="18.5" r="2"/>'
        '<path d="M6.5 6.9l4 3.7"/><path d="M13.5 13.4l4 3.7"/>'
    ),
    "portfolio": (
        '<path d="M20.5 13.5A8.5 8.5 0 1 1 10.5 3.5v10z"/>'
        '<path d="M14 3.1A8.5 8.5 0 0 1 20.9 10H14z"/>'
    ),
    "agent": (
        '<circle cx="12" cy="12" r="3"/>'
        '<circle cx="4" cy="6" r="1.6"/><circle cx="20" cy="6" r="1.6"/><circle cx="12" cy="20" r="1.6"/>'
        '<line x1="9.6" y1="10.2" x2="5.2" y2="7.1"/><line x1="14.4" y1="10.2" x2="18.8" y2="7.1"/>'
        '<line x1="12" y1="15" x2="12" y2="18.2"/>'
    ),
    "market": (
        '<line x1="3" y1="20" x2="21" y2="20"/>'
        '<line x1="6" y1="19" x2="6" y2="10"/><line x1="12" y1="19" x2="12" y2="5"/>'
        '<line x1="18" y1="19" x2="18" y2="13"/>'
    ),
    "lab": (
        '<path d="M9 3h6"/>'
        '<path d="M10 3v5.7L4.7 17.9a2 2 0 0 0 1.73 3h11.14a2 2 0 0 0 1.73-3L14 8.7V3"/>'
        '<line x1="8.2" y1="14" x2="15.8" y2="14"/>'
    ),
    "paper": '<polyline points="2,13 7,13 10,5 14,20 17,13 22,13"/>',
    "learn": (
        '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H12v17H6.5A2.5 2.5 0 0 0 4 22.5"/>'
        '<path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H12v17h5.5a2.5 2.5 0 0 1 2.5 2.5"/>'
        '<path d="M4 5.5v14"/><path d="M20 5.5v14"/>'
    ),
    "settings": (
        '<line x1="4" y1="6" x2="20" y2="6"/><circle cx="9" cy="6" r="2"/>'
        '<line x1="4" y1="12" x2="20" y2="12"/><circle cx="15" cy="12" r="2"/>'
        '<line x1="4" y1="18" x2="20" y2="18"/><circle cx="7" cy="18" r="2"/>'
    ),
}


def _icon_mask_data_uri(inner_paths: str) -> str:
    """A `mask-image` data URI for one icon -- the actual stroke color in
    this SVG is irrelevant (a mask uses only the rendered shape's alpha), so
    it is fixed at `black`; the CSS consumer paints it via
    `background-color: currentColor` instead, which is what makes the icon
    follow the row's own hover/selected text color automatically."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
        f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{inner_paths}</svg>'
    )
    return f'url("data:image/svg+xml,{_urlquote(svg)}")'


_NAV_ICON_CSS = "\n".join(
    f'[data-testid="stSidebar"] [class*="st-key-navrow-{name}"] button::before {{\n'
    f'  content: ""; position: absolute; left: 0.85rem; top: 50%; transform: translateY(-50%);\n'
    f'  width: 19px; height: 19px; background-color: currentColor;\n'
    f'  -webkit-mask-image: {_icon_mask_data_uri(paths)}; mask-image: {_icon_mask_data_uri(paths)};\n'
    f'  -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;\n'
    f'  -webkit-mask-size: contain; mask-size: contain; -webkit-mask-position: center; mask-position: center;\n'
    f'}}'
    for name, paths in _ICON_PATHS.items()
)


_CSS = f"""
<style>
:root {{
  --aa-bg: {palette.BG};
  --aa-bg-sidebar: {palette.BG_SIDEBAR};
  --aa-card: {palette.CARD_BG};
  --aa-card-raised: {palette.CARD_BG_RAISED};
  --aa-border: {palette.BORDER};
  --aa-border-soft: {palette.BORDER_SOFT};
  --aa-text: {palette.TEXT_PRIMARY};
  --aa-text-secondary: {palette.TEXT_SECONDARY};
  --aa-text-muted: {palette.TEXT_MUTED};
  --aa-blue: {palette.BLUE};
}}

/* -- base typography --------------------------------------------------------
   clamp(min, viewport-relative, max) instead of a fixed px/rem value: a body
   size that is legible at 150% browser zoom on a laptop-width window without
   becoming oversized at 100% zoom on a wide monitor -- see the "RESPONSIVE
   LAYOUT SYSTEM" block below for why vw-based sizing tracks zoom at all
   (zoom shrinks the same CSS-pixel viewport these units read). ------------ */
html, body, [data-testid="stAppViewContainer"] {{ font-size: clamp(14px, 0.3vw + 13.3px, 16px); }}
[data-testid="stMarkdownContainer"] p {{ font-size: clamp(0.88rem, 0.15vw + 0.85rem, 0.95rem); line-height: 1.55; }}
[data-testid="stCaptionContainer"] {{ font-size: clamp(0.76rem, 0.1vw + 0.74rem, 0.82rem); }}
h1 {{ font-size: clamp(1.5rem, 1.1vw + 1.3rem, 1.95rem) !important; font-weight: 800 !important; }}
h2 {{ font-size: clamp(1.3rem, 0.8vw + 1.15rem, 1.65rem) !important; font-weight: 800 !important; }}
h3 {{ font-size: clamp(1.05rem, 0.45vw + 0.95rem, 1.25rem) !important; font-weight: 700 !important; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: var(--aa-text); }}

/* -- page density -------------------------------------------------------- */
[data-testid="stMainBlockContainer"] {{ padding-top: 0.9rem; padding-bottom: 2rem; max-width: 100%; }}
[data-testid="stAppViewContainer"] {{ background-color: var(--aa-bg); }}
/* stHeader is deliberately NOT display:none (a prior version did this, which
   broke the sidebar collapse->restore affordance: Streamlit renders the
   "expand sidebar" button as a descendant of stHeader/stToolbar only while
   the sidebar is collapsed, so hiding stHeader unconditionally hid it right
   when it was needed and left no way to bring the sidebar back). stHeader is
   already position:absolute and empty except for the deploy button
   (hidden below) and that conditional expand control, so leaving it alone
   costs no layout space and fixes the bug. */
[data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stExpandSidebarButton"] {{ visibility: visible !important; opacity: 1 !important; }}
div[data-testid="stElementContainer"] {{ margin-bottom: 0.15rem; }}
div[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}

/* -- sidebar shell ---------------------------------------------------------
   Sidebar IA pass: the shared sidebar is now a fully custom nav (see
   `render_sidebar_nav`) -- `st.navigation(..., position="hidden")` supplies
   real multipage routing with no visible native list of its own, so every
   rule below targets stable, semantic `st.container(key=...)` classes this
   module itself assigns (`st-key-navrow-<name>`/`st-key-navsub-<name>-
   <sub>`), never Streamlit's native `stSidebarNav*` markup or a fragile
   position-based selector. */
section[data-testid="stSidebar"] {{
  background-color: var(--aa-bg-sidebar); border-right: 1px solid var(--aa-border-soft);
  min-width: 264px !important; max-width: 264px !important;
}}
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"] {{ padding-top: 0.7rem; padding-left: 0.85rem; padding-right: 0.85rem; }}

.aa-nav-brand {{
  font-size: 0.76rem; font-weight: 700; letter-spacing: 0.09em; color: var(--aa-text-muted);
  text-transform: uppercase; margin: 0.3rem 0.5rem 0.1rem 0.5rem;
}}
.aa-nav-tagline {{
  font-size: 0.62rem; font-weight: 400; letter-spacing: 0.01em; color: var(--aa-text-muted);
  text-transform: none; opacity: 0.75; margin: 0 0.5rem 0.8rem 0.5rem;
}}
.aa-nav-sep {{ border-top: 1px solid var(--aa-border-soft); margin: 0.7rem 0.4rem 0.6rem 0.4rem; }}

/* Primary destinations (Agent/Market/Research/Paper): 44-48px rows,
   stronger type, a restrained selected surface -- reuses the SAME
   primary/secondary button theming every other sidebar button on this
   platform already had (below), only sized/positioned for a nav row with
   room for its own icon. */
[data-testid="stSidebar"] .stButton button {{ width: 100%; text-align: left; }}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  background-color: {palette.BLUE_SOFT}; color: var(--aa-text); border: 1px solid {palette.BLUE_BORDER};
  box-shadow: inset 3px 0 0 var(--aa-blue); justify-content: flex-start; font-weight: 700;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
  background-color: transparent; border: 1px solid transparent; color: var(--aa-text-secondary);
  justify-content: flex-start; font-weight: 500;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
  background-color: var(--aa-card-raised); color: var(--aa-text); border-color: var(--aa-border);
}}
[data-testid="stSidebar"] [class*="st-key-navrow-"] button {{
  position: relative; height: 46px; font-size: 1.02rem; font-weight: 650;
  border-radius: 9px; padding-left: 2.6rem !important;
}}
[data-testid="stSidebar"] [class*="st-key-navrow-"] [data-testid="stBaseButton-primary"] p {{ font-weight: 800; }}
{_NAV_ICON_CSS}

/* Research/Paper/Learn's own internal-view sub-rows: indented, smaller, muted --
   progressive disclosure (section 7), only ever rendered while that group
   is the active top-level page. */
[data-testid="stSidebar"] [class*="st-key-navsub-"] button {{
  height: 34px; font-size: 0.86rem; font-weight: 500; border-radius: 7px;
  padding-left: 2.85rem !important; color: var(--aa-text-muted);
}}
[data-testid="stSidebar"] [class*="st-key-navsub-"] [data-testid="stBaseButton-primary"] {{
  box-shadow: inset 2px 0 0 var(--aa-blue);
}}
[data-testid="stSidebar"] [class*="st-key-navsub-"] [data-testid="stBaseButton-primary"] p {{
  color: var(--aa-text) !important; font-weight: 700;
}}

/* Settings -- secondary to the five product destinations above it: smaller
   type, normal weight, muted color; the `.aa-nav-sep` rule above already
   gives it a thin top rule for separation instead of a second bold section
   header (there is only ever one, "Agentic Alpha", at the very top). */
[data-testid="stSidebar"] [class*="st-key-navrow-settings"] button {{
  height: 36px; font-size: 0.88rem; font-weight: 500; color: var(--aa-text-muted);
}}
[data-testid="stSidebar"] [class*="st-key-navrow-settings"] [data-testid="stBaseButton-primary"] p {{
  color: var(--aa-text) !important; font-weight: 700;
}}

/* -- brand header block (custom, in-page) --------------------------------- */
.aa-topbar {{
  display: flex; align-items: center; justify-content: space-between;
  min-height: 56px; padding: 0.5rem 0.2rem; margin-bottom: 0.7rem;
  border-bottom: 1px solid var(--aa-border-soft);
}}
.aa-brand {{ display: flex; align-items: center; gap: 0.75rem; }}
.aa-brand-mark {{
  width: 38px; height: 38px; flex-shrink: 0; border-radius: 9px; display: flex; align-items: center;
  justify-content: center; background: {palette.BG_SIDEBAR}; border: 1px solid var(--aa-border);
}}
.aa-brand-mark svg {{ width: 26px; height: 26px; }}
.aa-brand-name {{ font-size: 1.22rem; font-weight: 800; color: var(--aa-text); line-height: 1.15; }}
.aa-brand-subtitle {{ font-size: 0.82rem; color: var(--aa-text-secondary); font-weight: 500; }}
.aa-pillbar {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; justify-content: flex-end; }}
.aa-pill {{ display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: var(--aa-text-secondary); }}
.aa-pill-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.aa-pill-label {{ white-space: nowrap; }}
.aa-topbar-time {{ font-size: 0.78rem; color: var(--aa-text-muted); text-align: right; margin-top: 0.25rem; }}

/* -- optional IBKR delayed market-context strip (Agent / Dashboard) -------- */
.aa-marketstrip {{
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
  padding: 0.5rem 0.2rem; margin-bottom: 0.6rem; border-bottom: 1px solid var(--aa-border-soft);
  font-size: 0.84rem;
}}
.aa-marketstrip-ticker {{ font-weight: 800; color: var(--aa-text); font-size: 0.95rem; }}
.aa-marketstrip-item {{ color: var(--aa-text-secondary); }}
.aa-marketstrip-item b {{ color: var(--aa-text); }}
.aa-marketstrip-ts {{ color: var(--aa-text-muted); font-size: 0.76rem; margin-left: auto; }}
.aa-marketstrip-disconnected {{ color: var(--aa-text-muted); }}
.aa-marketstrip-waiting {{ color: var(--aa-text-muted); }}
.aa-marketstrip-unavailable {{ color: var(--aa-text-muted); }}
.aa-marketstrip-hint {{ font-size: 0.78rem; color: var(--aa-text-muted); }}

/* -- market ticker (Overview) ---------------------------------------------- */
.aa-ticker-row {{ display: flex; align-items: baseline; gap: 0.75rem; }}
.aa-ticker {{ font-size: 2.05rem; font-weight: 800; color: var(--aa-text); letter-spacing: -0.02em; line-height: 1; }}
.aa-ticker-name {{ font-size: 0.98rem; color: var(--aa-text-secondary); font-weight: 500; }}

/* -- HOME TERMINAL: compact ticker cards (market_home.render_ticker_strip) - */
.aa-ticker-card-root {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; color: var(--aa-text-muted);
  text-transform: uppercase; }}
.aa-ticker-card-last {{ font-size: 1.28rem; font-weight: 800; color: var(--aa-text); margin-top: 0.15rem;
  line-height: 1.15; overflow-wrap: normal; }}
.aa-ticker-card-change {{ font-size: 0.84rem; font-weight: 700; margin-top: 0.1rem; }}

/* -- HOME TERMINAL: Selected Market hero (market_home.render_selected_market_hero) - */
.aa-hero-market {{ font-size: 1.0rem; font-weight: 600; color: var(--aa-text-secondary); }}
.aa-hero-price {{ font-size: clamp(2.0rem, 2.2vw + 1.5rem, 2.8rem); font-weight: 800; color: var(--aa-text);
  letter-spacing: -0.02em; line-height: 1.15; margin-top: 0.1rem; overflow-wrap: normal; }}
.aa-hero-change {{ font-size: 1.1rem; font-weight: 700; margin-top: 0.15rem; }}
.aa-hero-contract-label {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em; color: var(--aa-text-muted);
  text-transform: uppercase; }}
.aa-hero-contract-value {{ font-size: 1.05rem; font-weight: 700; color: var(--aa-text); margin-top: 0.2rem; }}

/* -- cards ------------------------------------------------------------------ */
div.stVerticalBlock[class*="st-key-card-"] {{
  background-color: var(--aa-card) !important; border: 1px solid var(--aa-border) !important;
  border-radius: 9px !important; gap: 0.3rem !important;
  padding: clamp(0.75rem, 0.4vw + 0.65rem, 0.95rem) clamp(0.85rem, 0.5vw + 0.75rem, 1.05rem) !important;
}}

.aa-section-title {{ font-size: 1.02rem; font-weight: 700; color: var(--aa-text); margin: 1rem 0 0.15rem 0; }}
.aa-section-subtitle {{ font-size: 0.82rem; color: var(--aa-text-muted); margin-bottom: 0.55rem; }}

/* -- Agent conversation composer bar (Agent Experience Consolidation
   campaign, sections 15-16) -- visually primary (full width, clear border/
   shadow contrast, a ~48px input), and STICKY to the bottom of Agent's own
   content column so it stays reachable after a long Claude answer or an
   expanded Research Workspace -- `position: sticky` inside this page's own
   block, never a fixed/absolute overlay that could cover content or the
   sidebar. There is exactly one such container per render (`agent.py`'s
   own composer-rendering function is the only caller of
   `components.card("agent-composer-bar")`), so this never doubles up.
   NOTE: this comment deliberately never spells out the composer's own
   on-page label text -- that literal phrase would otherwise appear inside
   this injected stylesheet's `st.markdown` block, ahead of the label's own
   real render further down the page, which breaks any test asserting on
   where that phrase FIRST appears (see `test_market_home.py`'s own
   "aa-prov-label" precedent for this exact class of false-positive). ---- */
div.stVerticalBlock[class*="st-key-card-agent-composer-bar"] {{
  position: sticky; bottom: 0.6rem; z-index: 5;
  border-color: {palette.BLUE_BORDER} !important;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}}
div.stVerticalBlock[class*="st-key-card-agent-composer-bar"] [data-testid="stTextInput"] input {{
  height: 3rem; font-size: 0.98rem; padding: 0 0.9rem;
}}
div.stVerticalBlock[class*="st-key-card-agent-composer-bar"] [data-testid="stBaseButton-primary"] {{
  height: 3rem; font-size: 0.98rem; font-weight: 700;
}}

/* `overflow-wrap: normal` (the CSS default -- stated explicitly so it is
   never accidentally overridden to `anywhere`/`break-word`) is deliberate on
   both of these: a value with no natural break point ("$163", "REJECT") has
   nothing to wrap at, so the metric-row grid's own `min-width:auto` default
   simply refuses to shrink that column below it -- never a forced mid-token
   break. A genuinely long value that DOES carry a natural break point (a
   date range, a long identifier with dashes) wraps there instead, which is
   correct; `white-space: nowrap` was tried here and reverted -- it stopped
   in-word splitting but made a too-long unbreakable line bleed sideways into
   the next card instead, which is worse. */
.aa-metric-label {{ font-size: clamp(0.68rem, 0.1vw + 0.65rem, 0.72rem); color: var(--aa-text-muted); text-transform: uppercase;
  letter-spacing: 0.03em; font-weight: 700; overflow-wrap: normal; }}
.aa-metric-value {{ font-size: clamp(1.15rem, 1.1vw + 0.85rem, 1.62rem); font-weight: 800; color: var(--aa-text);
  margin-top: 0.2rem; line-height: 1.15; overflow-wrap: normal; }}
.aa-metric-value.aa-muted {{ color: var(--aa-text-muted); font-size: clamp(0.95rem, 0.7vw + 0.75rem, 1.3rem); }}
.aa-metric-sub {{ font-size: 0.74rem; margin-top: 0.2rem; color: var(--aa-text-muted); line-height: 1.3; cursor: help; }}

.aa-gate-title {{ font-size: 0.9rem; font-weight: 700; color: var(--aa-text); }}
.aa-gate-stat {{ font-size: 0.82rem; color: var(--aa-text-secondary); margin-top: 0.3rem; }}
.aa-gate-note {{ font-size: 0.76rem; color: var(--aa-text-muted); margin-top: 0.2rem; }}

.aa-badge {{
  display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.74rem; font-weight: 700;
  padding: 0.16rem 0.6rem; border-radius: 999px; border: 1px solid; letter-spacing: 0.02em;
  white-space: nowrap;
}}

.aa-empty {{ text-align: center; padding: 1rem 0.5rem; }}
.aa-empty-title {{ font-size: 0.86rem; font-weight: 700; color: var(--aa-text-secondary); margin-bottom: 0.5rem; }}
.aa-empty-badge {{ margin-bottom: 0.5rem; }}
.aa-empty-reason {{ font-size: 0.76rem; color: var(--aa-text-muted); max-width: 32rem; margin: 0 auto; line-height: 1.4; }}

.aa-prov-row {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.3rem 1rem; font-size: 0.8rem; padding: 0.3rem 0; border-bottom: 1px dashed var(--aa-border-soft); }}
.aa-prov-label {{ color: var(--aa-text-muted); flex: 0 0 auto; }}
.aa-prov-value {{ color: var(--aa-text); font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.76rem;
  text-align: right; word-break: break-all; flex: 1 1 60%; min-width: 0; }}

.aa-activity-row {{ display: flex; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid var(--aa-border-soft); align-items: flex-start; }}
.aa-activity-icon {{ width: 7px; height: 7px; border-radius: 50%; margin-top: 0.5rem; flex-shrink: 0; background: {palette.BLUE}; }}
.aa-activity-title {{ font-size: 0.84rem; font-weight: 600; color: var(--aa-text); }}
.aa-activity-detail {{ font-size: 0.76rem; color: var(--aa-text-muted); }}
.aa-activity-time {{ font-size: 0.72rem; color: var(--aa-text-muted); white-space: nowrap; }}

.aa-status-row {{ display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.84rem; padding: 0.4rem 0;
  border-bottom: 1px solid var(--aa-border-soft); flex-wrap: wrap; }}
.aa-status-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 0.32rem; }}
.aa-status-label {{ color: var(--aa-text); font-weight: 600; flex: 1 1 9rem; }}
.aa-status-detail {{ color: var(--aa-text-muted); font-size: 0.76rem; flex: 1 1 100%; margin-left: 1.15rem;
  overflow-wrap: anywhere; }}

.aa-tag {{ display: inline-block; font-size: 0.72rem; font-weight: 600; color: {palette.BLUE};
  background: {palette.BLUE_SOFT}; border: 1px solid {palette.BLUE_BORDER}; border-radius: 999px;
  padding: 0.12rem 0.55rem; margin: 0.2rem 0.25rem 0.2rem 0; }}
.aa-tag-row {{ margin-bottom: 0.5rem; }}

/* -- tabs ------------------------------------------------------------------ */
[data-testid="stTab"] p {{ font-size: 0.88rem; color: var(--aa-text-secondary); }}
[data-testid="stTab"][aria-selected="true"] p {{ color: var(--aa-text); font-weight: 700; }}

/* -- misc widgets ------------------------------------------------------------ */
[data-testid="stMetricValue"] {{ font-size: clamp(1.1rem, 1vw + 0.8rem, 1.5rem) !important; overflow-wrap: normal; }}
[data-testid="stMetricLabel"] {{ font-size: clamp(0.76rem, 0.15vw + 0.72rem, 0.82rem); }}
[data-testid="stSelectbox"] label, [data-testid="stTextInput"] label {{ font-size: 0.85rem; }}

/* comfortable textareas regardless of the small `height=` px a page passes --
   never solved by shrinking the font to fit. */
[data-testid="stTextArea"] textarea {{ min-height: 92px !important; line-height: 1.5 !important; }}

/* ============================================================================
   RESEARCH THREAD WORKSPACE (News / Research / Portfolio / Learn)
   ----------------------------------------------------------------------------
   One quiet visual language for the research journey: muted eyebrows, one
   strong title per screen, chains of economic states as small nodes, and a
   progress stepper whose only accent is the CURRENT step. Path types carry a
   thin left rule in a categorical (never a verdict) color.
   ============================================================================ */
.aa-eyebrow {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--aa-text-muted); }}
.aa-page-title {{ font-size: clamp(1.45rem, 1vw + 1.2rem, 1.85rem); font-weight: 800; color: var(--aa-text);
  letter-spacing: -0.015em; line-height: 1.2; margin: 0.1rem 0 0.2rem 0; }}
.aa-page-lede {{ font-size: 0.92rem; color: var(--aa-text-secondary); max-width: 52rem; line-height: 1.5;
  margin-bottom: 0.4rem; }}
.aa-thread-title {{ font-size: clamp(1.35rem, 1vw + 1.1rem, 1.75rem); font-weight: 800; color: var(--aa-text);
  letter-spacing: -0.015em; line-height: 1.2; margin: 0.15rem 0 0.2rem 0; }}
.aa-thread-sub {{ font-size: 0.86rem; color: var(--aa-text-secondary); line-height: 1.45; }}
.aa-thread-sub b {{ color: var(--aa-text); font-weight: 600; }}

/* -- Research Setup strip -- */
.aa-setup-strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.45rem; }}
.aa-setup-strip .aa-eyebrow {{ margin-right: 0.25rem; }}
.aa-setup-chip {{ display: inline-flex; gap: 0.35rem; align-items: baseline; font-size: 0.78rem; color: var(--aa-text);
  background: var(--aa-card); border: 1px solid var(--aa-border); border-radius: 999px; padding: 0.14rem 0.62rem; }}
.aa-setup-k {{ color: var(--aa-text-muted); font-weight: 600; }}
.aa-setup-default {{ font-size: 0.68rem; font-weight: 600; color: {palette.AMBER}; border: 1px solid {palette.AMBER}55;
  border-radius: 999px; padding: 0.05rem 0.45rem; margin-right: 0.15rem; cursor: help; }}

/* -- progress stepper: a hairline track behind seven pill buttons -- */
div.stVerticalBlock[class*="st-key-thread-stepper"] {{ position: relative; margin: 0.35rem 0 0.2rem 0; }}
div.stVerticalBlock[class*="st-key-thread-stepper"]::before {{
  content: ""; position: absolute; left: 1.5rem; right: 1.5rem; top: 50%; height: 1px;
  background: var(--aa-border); z-index: 0;
}}
[class*="st-key-step-"] button {{ width: 100%; min-height: 38px; border-radius: 999px !important;
  font-size: 0.84rem; position: relative; z-index: 1; }}
[class*="st-key-step-"] button p {{ color: inherit !important; font-weight: inherit !important; }}
[class*="st-key-step-done-"] button {{ background: var(--aa-card) !important; border: 1px solid var(--aa-border) !important;
  color: var(--aa-text-secondary) !important; font-weight: 600; }}
[class*="st-key-step-done-"] button:hover {{ border-color: {palette.BLUE_BORDER} !important; color: var(--aa-text) !important; }}
[class*="st-key-step-done-"] button [data-testid="stIconMaterial"] {{ color: {palette.BLUE}; }}
[class*="st-key-step-current-"] button {{ background: {palette.BLUE_SOFT} !important; border: 1px solid {palette.BLUE} !important;
  color: var(--aa-text) !important; font-weight: 800; }}
[class*="st-key-step-current-"] button [data-testid="stIconMaterial"] {{ color: {palette.BLUE}; }}
[class*="st-key-step-todo-"] button {{ background: var(--aa-bg) !important; border: 1px dashed var(--aa-border) !important;
  color: var(--aa-text-muted) !important; font-weight: 500; }}
[class*="st-key-step-todo-"] button:disabled {{ opacity: 1; cursor: not-allowed; }}
[class*="st-key-step-todo-"] button:not(:disabled):hover {{ border-style: solid !important; color: var(--aa-text) !important; }}

/* -- step heading -- */
.aa-step-head {{ display: flex; align-items: baseline; gap: 0.65rem; margin: 0.5rem 0 0 0; flex-wrap: wrap; }}
.aa-step-num {{ font-size: 0.7rem; font-weight: 700; color: var(--aa-text-muted); letter-spacing: 0.09em;
  text-transform: uppercase; }}
.aa-step-title {{ font-size: 1.32rem; font-weight: 800; color: var(--aa-text); letter-spacing: -0.01em; }}
.aa-step-q {{ font-size: 0.93rem; color: var(--aa-text-secondary); margin: 0.05rem 0 0.55rem 0; }}
.aa-subhead {{ font-size: 0.98rem; font-weight: 700; color: var(--aa-text); margin: 1rem 0 0.1rem 0; }}
.aa-subnote {{ font-size: 0.8rem; color: var(--aa-text-muted); margin-bottom: 0.45rem; line-height: 1.45; }}

/* -- research context panel -- */
.aa-ctx {{ display: flex; flex-direction: column; }}
.aa-ctx-row {{ padding: 0.45rem 0; border-bottom: 1px solid var(--aa-border-soft); }}
.aa-ctx-row:last-child {{ border-bottom: none; }}
.aa-ctx-label {{ font-size: 0.64rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--aa-text-muted); }}
.aa-ctx-value {{ font-size: 0.85rem; color: var(--aa-text); margin-top: 0.12rem; line-height: 1.4; overflow-wrap: anywhere; }}
.aa-ctx-value .aa-dim {{ color: var(--aa-text-muted); }}

/* -- chains of economic states (lineage / signal paths) -- */
.aa-chain {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.32rem; margin: 0.3rem 0 0.2rem 0; }}
.aa-node {{ font-size: 0.79rem; color: var(--aa-text); background: var(--aa-card-raised); border: 1px solid var(--aa-border);
  border-radius: 6px; padding: 0.14rem 0.5rem; line-height: 1.3; }}
.aa-node-start {{ border-color: {palette.BLUE_BORDER}; background: {palette.BLUE_SOFT}; }}
.aa-node-end {{ font-weight: 700; border-color: var(--aa-text-muted); }}
.aa-node-market {{ font-weight: 700; border-style: dashed; }}
.aa-arrow {{ color: var(--aa-text-muted); font-size: 0.78rem; }}
.aa-kind {{ font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }}

/* -- small label/value grid -- */
.aa-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 0.4rem 0.9rem; margin-top: 0.35rem; }}
.aa-fact-k {{ font-size: 0.63rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--aa-text-muted); }}
.aa-fact-v {{ font-size: 0.85rem; color: var(--aa-text); margin-top: 0.1rem; line-height: 1.3; }}
.aa-fact-v.aa-dim {{ color: var(--aa-text-muted); }}

/* -- path cards: a thin rule in the path type's categorical color -- */
div.stVerticalBlock[class*="st-key-card-path-direct-"] {{ border-left: 3px solid {palette.CATEGORICAL[0]} !important; }}
div.stVerticalBlock[class*="st-key-card-path-supply_chain-"] {{ border-left: 3px solid {palette.CATEGORICAL[2]} !important; }}
div.stVerticalBlock[class*="st-key-card-path-cross_sector-"] {{ border-left: 3px solid {palette.CATEGORICAL[1]} !important; }}
div.stVerticalBlock[class*="st-key-card-path-"][class*="-followed-"] {{ background: {palette.CARD_BG_RAISED} !important;
  box-shadow: 0 0 0 1px {palette.BLUE_BORDER}; }}
div.stVerticalBlock[class*="st-key-card-mx-"][class*="-focus-"] {{ box-shadow: 0 0 0 1px {palette.BLUE}; }}

/* -- news cards -- */
.aa-news-meta {{ font-size: 0.74rem; color: var(--aa-text-muted); display: flex; gap: 0.45rem; flex-wrap: wrap;
  align-items: center; }}
.aa-news-meta .aa-dot-sep {{ opacity: 0.6; }}
.aa-news-new {{ color: {palette.BLUE}; font-weight: 700; }}
.aa-news-headline {{ font-size: 1.04rem; font-weight: 700; color: var(--aa-text); line-height: 1.38; margin: 0.18rem 0 0.15rem 0; }}
.aa-news-summary {{ font-size: 0.84rem; color: var(--aa-text-secondary); line-height: 1.5; }}

/* -- impact meters: level as filled segments (a triage level, not a return) -- */
.aa-impact-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.15rem 0; }}
.aa-impact {{ display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.78rem; padding: 0.2rem 0.62rem;
  border-radius: 8px; border: 1px solid var(--aa-border); background: var(--aa-card-raised); color: var(--aa-text); }}
.aa-impact b {{ font-weight: 700; letter-spacing: 0.02em; }}
.aa-impact-out {{ background: transparent; border-style: dashed; color: var(--aa-text-muted); }}
.aa-meter {{ display: inline-flex; gap: 2px; align-items: flex-end; }}
.aa-meter i {{ width: 4px; height: 11px; border-radius: 1.5px; background: var(--aa-border); display: inline-block; }}
.aa-meter i.on {{ background: {palette.BLUE}; }}

/* -- data availability rows (Measure it) -- */
.aa-measure {{ display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 1fr); gap: 0.2rem 0.8rem;
  padding: 0.34rem 0; border-bottom: 1px dashed var(--aa-border-soft); font-size: 0.82rem; align-items: center; }}
.aa-measure-head {{ font-size: 0.63rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--aa-text-muted); border-bottom: 1px solid var(--aa-border-soft); }}
.aa-measure-src {{ font-size: 0.72rem; color: var(--aa-text-muted); font-weight: 400; }}
.aa-state {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.76rem; color: var(--aa-text-secondary); }}
.aa-state::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--aa-state, {palette.GREY}); }}
.aa-state-yes {{ --aa-state: {palette.GREEN}; color: var(--aa-text); }}
.aa-state-part {{ --aa-state: {palette.AMBER}; }}
.aa-state-no {{ --aa-state: {palette.GREY}; color: var(--aa-text-muted); }}

/* Streamlit pulls a markdown block up by -1rem (it assumes a trailing <p>
   margin); a block ending in one of these custom layouts has no such margin,
   so the pull would clip its last row under the next element. */
[data-testid="stMarkdownContainer"]:has(> div[class^="aa-"]:last-child) {{ margin-bottom: 0 !important; }}

/* -- inline "i" tooltip: hover or keyboard-focus shows what a thing means -- */
.aa-info {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 15px;
  height: 15px; border-radius: 50%; border: 1px solid var(--aa-text-muted); color: var(--aa-text-muted);
  font-size: 0.6rem; font-weight: 800; font-style: normal; margin-left: 0.4rem; cursor: help; vertical-align: 0.1em;
  text-transform: none; letter-spacing: 0; line-height: 1; }}
.aa-info:hover, .aa-info:focus {{ color: var(--aa-text); border-color: var(--aa-text-secondary); outline: none; }}
.aa-info::after {{ content: attr(data-tip); position: absolute; left: -0.6rem; top: calc(100% + 8px); width: max-content;
  max-width: 330px; white-space: normal; background: {palette.CARD_BG_RAISED}; color: var(--aa-text);
  border: 1px solid var(--aa-border); border-radius: 8px; padding: 0.55rem 0.7rem; font-size: 0.78rem; font-weight: 400;
  line-height: 1.45; box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45); opacity: 0; pointer-events: none;
  transition: opacity 0.12s ease; z-index: 60; }}
.aa-info:hover::after, .aa-info:focus::after {{ opacity: 1; }}

/* -- one-line meta row inside cards -- */
.aa-meta {{ display: flex; flex-wrap: wrap; gap: 0.25rem 0.9rem; margin: 0.35rem 0 0.1rem 0; font-size: 0.8rem;
  color: var(--aa-text); }}
.aa-meta span b {{ font-weight: 600; color: var(--aa-text-muted); font-size: 0.66rem; letter-spacing: 0.07em;
  text-transform: uppercase; margin-right: 0.3rem; }}

/* -- Market: the selected market, market-first -- */
.aa-market-title {{ font-size: clamp(1.3rem, 0.9vw + 1.1rem, 1.65rem); font-weight: 800; color: var(--aa-text);
  letter-spacing: -0.015em; line-height: 1.2; margin: 0.15rem 0 0.35rem 0; }}
.aa-market-line {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.5rem; font-size: 0.84rem;
  color: var(--aa-text-secondary); }}
.aa-context-strip {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.3rem 0.6rem; padding: 0.55rem 0.8rem;
  border: 1px solid var(--aa-border); border-radius: 9px; background: var(--aa-card); margin: 0.3rem 0 0.2rem 0; }}
.aa-context-strip b {{ color: var(--aa-text); }}

/* -- continuation bar -- */
div.stVerticalBlock[class*="st-key-thread-nav"] {{ border-top: 1px solid var(--aa-border-soft); padding-top: 0.75rem;
  margin-top: 1.2rem; }}

/* -- sidebar: the small "Tools" group below the research journey -- */
.aa-nav-group {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--aa-text-muted); margin: 0.2rem 0.55rem 0.25rem 0.55rem; opacity: 0.85; }}
[data-testid="stSidebar"] [class*="st-key-navtools"] [class*="st-key-navrow-"] button {{
  height: 38px; font-size: 0.9rem; font-weight: 500;
}}
[data-testid="stSidebar"] [class*="st-key-navtools"] [class*="st-key-navrow-"] [data-testid="stBaseButton-secondary"] {{
  color: var(--aa-text-muted);
}}

@media (max-width: 899px) {{
  [class*="st-key-thread-stepper"] [data-testid="stColumn"] {{ min-width: 92px !important; flex: 1 1 92px !important; }}
  div.stVerticalBlock[class*="st-key-thread-stepper"]::before {{ display: none; }}
}}

/* ============================================================================
   RESPONSIVE LAYOUT SYSTEM (release: responsive layout hardening)
   ----------------------------------------------------------------------------
   Breakpoints -- a browser zoom step narrows this same CSS-pixel viewport
   exactly like a smaller monitor would, so these bands cover both axes:

     LARGE    >= 1500px : full main-rail ratio (MAIN_RAIL_RATIO), metric
                          grids at their widest.
     DESKTOP  1200-1499 : the right rail gets a fixed, wider relative share
                          (it would otherwise shrink in lockstep with the
                          main column and become an unusable sliver).
     COMPACT   900-1199 : the main/rail split stacks -- rail moves BELOW the
                          main column instead of squeezing.
     NARROW    < 900px  : every remaining multi-column row (control rows,
                          chart pairs) wraps onto multiple lines.

   Streamlit's own `st.columns(..., wrap=True)` already stacks columns at
   <=640px; these bands sit above that and are implemented purely in CSS
   against stable `data-testid` attributes plus this project's own
   `st.container(key=...)` scoping classes (`layout.main_rail_columns`,
   `components.metric_row`, `components.gate_row`) -- never against a
   Streamlit-generated `st-emotion-cache-*` hash.
   ============================================================================ */

/* -- metric / gate card grid --------------------------------------------------
   Any row wrapped in `components.metric_row` / `components.gate_row` becomes a
   true CSS auto-fit grid: a value like "$163" or a label like "VERDICT" never
   breaks mid-word. Two things are required, not just `display:grid` on the
   row -- (1) Streamlit leaves an explicit percentage `width` on each column
   (left over from its flex layout, e.g. `calc(14.28% - 16px)`), which must be
   cleared for the grid track to actually govern sizing; (2) grid items default
   to `min-width:auto`, so a track never shrinks below its content's own
   minimum width -- the row wraps into more lines instead of splitting a value.
   The exact `> div[stLayoutWrapper] > div[stHorizontalBlock] > div[stColumn]`
   chain (verified against a real rendered Streamlit 1.63 DOM) keeps this
   scoped to THIS row's own columns, never a metric/gate row nested inside a
   main-rail column several levels deeper. -- */
[class*="st-key-metric-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"],
[class*="st-key-gate-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] {{
  display: grid !important;
  gap: 0.6rem !important;
}}
[class*="st-key-metric-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] {{
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
}}
[class*="st-key-gate-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] {{
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)) !important;
}}
[class*="st-key-metric-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
[class*="st-key-gate-row-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
  width: 100% !important; min-width: 0 !important;
}}

/* -- main/rail shell (`layout.main_rail_columns`) ----------------------------- */
/* The main column may always shrink: a wide unwrapped child (a long pill row,
   a table) must wrap inside it, never push the page wider than the viewport. */
[class*="st-key-main-rail-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"]
  > div[data-testid="stColumn"]:nth-of-type(1) {{ min-width: 0 !important; }}
@media (max-width: 1499px) and (min-width: 1200px) {{
  [class*="st-key-main-rail-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"]
    > div[data-testid="stColumn"]:nth-of-type(1) {{ flex: 2.1 1 0 !important; width: auto !important; }}
  [class*="st-key-main-rail-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"]
    > div[data-testid="stColumn"]:nth-of-type(2) {{
    flex: 1 1 0 !important; width: auto !important; min-width: 300px !important;
  }}
}}
@media (max-width: 1199px) {{
  [class*="st-key-main-rail-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] {{
    flex-direction: column !important;
  }}
  [class*="st-key-main-rail-"] > div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"]
    > div[data-testid="stColumn"] {{ width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }}
}}

/* -- narrow: every remaining multi-column control/chart row (selects, filter
   rows, chart pairs not scoped above) wraps rather than truncating -- */
@media (max-width: 899px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; row-gap: 0.6rem !important; }}
  [data-testid="stColumn"] {{ min-width: 240px !important; flex: 1 1 240px !important; }}
}}
</style>
"""


def inject_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def main_rail_columns(*, gap: str = "medium"):
    """The shared main-content / right-rail split every AI-agent-rail page
    (Agent, Dashboard, Research, Strategies) uses. Wrapped in one stable
    keyed container (`st-key-main-rail-shell`) so the injected responsive CSS
    can restructure the rail at DESKTOP/COMPACT widths (a fixed relative
    share, then a full stack below the main column) purely by targeting that
    key -- no page changes its own layout code as the breakpoint changes. The
    key is reused across pages on purpose: only one page's script executes
    per Streamlit run, so it never collides with itself."""
    with st.container(key="main-rail-shell"):
        return st.columns(MAIN_RAIL_RATIO, gap=gap)


def _topbar_html(brand_mark_svg: str, subtitle: str, right_side: str) -> str:
    """Pure string builder for `render_header`'s top bar -- kept separate
    and DEDENTED (no leading whitespace on any line) on purpose; see
    `test_topbar_html_never_has_a_blank_line` for why a blank line inside
    this block is a real rendering bug, not just a style nit (a prior
    version hit it when an interpolated value was empty). Sidebar IA pass
    (task spec section 12): the header is now brand name + ONE page-specific
    subtitle line -- the five backend-engineering pills, git commit, and
    build timestamp this block used to always render moved to Settings
    (Runtime / Data Providers / About); `right_side` now carries AT MOST one
    small, subtle indicator (`_health_attention_html`), and only when
    something actually needs attention (section 10C)."""
    return (
        f'<div class="aa-topbar">'
        f'<div class="aa-brand">'
        f'<div class="aa-brand-mark">{brand_mark_svg}</div>'
        f'<div>'
        f'<div class="aa-brand-name">AGENTIC ALPHA</div>'
        f'<div class="aa-brand-subtitle">{subtitle}</div>'
        f'</div>'
        f'</div>'
        f'<div>{right_side}</div>'
        f'</div>'
    )


def _health_attention_html() -> str:
    """The ONE small, subtle indicator `render_header` may show (task spec
    section 12) -- and ONLY when something genuinely needs attention (section
    10C: "healthy infrastructure should be quiet"). Never a permanent
    all-green status row; a healthy system renders nothing here at all. Full
    diagnostic detail always lives on Settings -> Runtime / Data Providers."""
    problems: list[str] = []
    try:
        engine = services.engine_provenance()
        if not engine["reference_cli_built"]:
            problems.append("C++ engine not built")
    except Exception:  # noqa: BLE001 -- header status is best-effort, never fails the page
        problems.append("engine status unavailable")
    try:
        catalog = services.catalog_summary()
        if catalog["n_rows"] == 0:
            problems.append("no market data catalogued")
    except Exception:  # noqa: BLE001
        problems.append("data catalog unavailable")
    try:
        services.registry_summary()
    except Exception:  # noqa: BLE001
        problems.append("registry unavailable")

    if not problems:
        return ""
    detail = "; ".join(problems)
    return (
        f'<span class="aa-badge" title="{detail} -- see Settings for detail." '
        f'style="color:{palette.state_color("WARN")};border-color:{palette.state_color("WARN")}66;'
        f'background:{palette.state_color("WARN")}1f;">{palette.state_icon("WARN")} Attention needed</span>'
    )


def render_header(*, subtitle: str = "") -> None:
    """The lightweight global header (task spec sections 10/12). Called once
    at the top of every page's `render()`, immediately after
    `render_sidebar_nav`, so the shell is identical everywhere: brand name +
    a page-specific one-line `subtitle` (each caller passes its own short
    purpose line), plus at most one small warning indicator when something
    needs attention. The five
    backend-engineering pills / git commit / build timestamp this header used
    to always render now live in Settings -> Runtime / Data Providers /
    About (task spec section 10A/10B) -- nothing was deleted, only relocated
    to where configuration/diagnostic detail belongs."""
    st.markdown(
        _topbar_html(_brand_mark_svg(), subtitle, _health_attention_html()), unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# shared sidebar -- WHERE DO I GO (task spec section 6), not everything the
# system knows. See `_ICON_PATHS`/`_NAV_ICON_CSS` above for the icon
# mechanism this renders through real `st.button` + `st.switch_page` rows.
# ---------------------------------------------------------------------------

#: Strategy Lab (the former Research page, url `lab`) keeps its own
#: drill-down views reachable from its landing; the sidebar only shows the
#: research journey's four destinations and a compact Tools group.
_LAB_SUBVIEWS: tuple[tuple[str, str], ...] = (
    ("strategies", "Strategies"), ("experiments", "Experiments"),
    ("validation", "Validation"), ("provenance", "Provenance"),
)
_PAPER_SUBVIEWS: tuple[tuple[str, str], ...] = (
    ("positions", "Positions"), ("execution", "Execution"),
    ("risk", "Risk"), ("monitoring", "Monitoring"),
)
#: Learn = personal research memory (My Alpha), what has been researched
#: before (Research Map), replication (Community), and the explanation layer
#: (Guides: concepts, strategies, failure autopsy, learning paths).
_LEARN_SUBVIEWS: tuple[tuple[str, str], ...] = (
    ("alpha_library", "My Alpha"), ("alpha_graph", "Research Map"),
    ("community", "Community"), ("guides", "Guides"),
)
_SETTINGS_SUBVIEWS: tuple[tuple[str, str], ...] = (
    ("providers", "Data Providers"), ("runtime", "Runtime"),
    ("claude", "Claude API"), ("about", "About"),
)

#: (nav key, label) -- the research journey, in order.
PRIMARY_NAV: tuple[tuple[str, str], ...] = (
    ("news", "News"), ("research", "Research"), ("market", "Market"), ("portfolio", "Portfolio"),
    ("learn", "Learn"),
)
#: (nav key, label) -- secondary tools.
TOOL_NAV: tuple[tuple[str, str], ...] = (
    ("agent", "Ask"), ("lab", "Strategy Lab"), ("paper", "Paper Trading"),
)
_TOOL_HELP = {
    "agent": "Ask the research assistant anything, or propose a strategy hypothesis directly.",
    "lab": "The single-strategy research lab: hypothesis, compile, C++ backtest, validation, and the full "
           "experiment record.",
    "paper": "Replay paper trading for validated strategies, with real risk limits.",
}
#: Which session key a group's sub-row deep-link writes (the page reads it).
_FOCUS_KEY = {"lab": "research_landing_focus", "paper": "paper_landing_focus", "learn": "learn_landing_focus",
              "settings": "settings_landing_focus"}
_SUBVIEWS = {"lab": _LAB_SUBVIEWS, "paper": _PAPER_SUBVIEWS, "learn": _LEARN_SUBVIEWS,
             "settings": _SETTINGS_SUBVIEWS}


def _nav_page_registry() -> dict[str, tuple]:
    """Lazy import -- every `views/*.py` module imports `layout` at module
    scope, so `layout` importing them back at module scope would cycle; this
    mirrors the existing lazy cross-page hand-off pattern already used
    elsewhere in this UI layer (e.g. `views/agent.py`'s
    `_view_research_details_button`)."""
    from alpha_agent.ui.views import (
        agent,
        learn,
        market,
        news,
        paper_trading,
        portfolio,
        research,
        system,
        workspace,
    )

    return {
        "news": (news.render, "news"),
        "research": (workspace.render, "research"),
        "portfolio": (portfolio.render, "portfolio"),
        "learn": (learn.render, "learn"),
        "agent": (agent.render, "agent"),
        "market": (market.render, "market"),
        "lab": (research.render, "lab"),
        "paper": (paper_trading.render, "paper-trading"),
        "settings": (system.render, "system"),
    }


def _nav_row(key: str, label: str, *, is_active: bool, help: str | None = None) -> bool:
    """One full-width nav row -- a real `st.button` (hover/focus/selected
    theming comes from the CSS above), carrying its icon as a CSS `::before`
    mask-image pseudo-element keyed to this row's own stable container
    class. Returns True exactly on the click that should navigate."""
    with st.container(key=f"navrow-{key}"):
        return st.button(label, key=f"navbtn-{key}", type=("primary" if is_active else "secondary"), width="stretch",
                         help=help)


def _nav_sub_row(group: str, sub_key: str, label: str, *, is_active: bool) -> bool:
    with st.container(key=f"navsub-{group}-{sub_key}"):
        return st.button(
            label, key=f"navsubbtn-{group}-{sub_key}", type=("primary" if is_active else "secondary"), width="stretch",
        )


def render_sidebar_nav(*, active: str | None, active_sub: str | None = None) -> None:
    """The shared sidebar's ONE job: where do I go. The research journey --
    News / Research / Market / Portfolio / Learn -- then a compact Tools group
    (Ask, Strategy Lab, Paper Trading), then Settings. Each row is a
    monochrome inline-SVG line icon driving real `st.switch_page` navigation
    via `st.navigation(..., position="hidden")` (see `app.py`).

    Learn / Strategy Lab / Paper / Settings are progressive-disclosure groups:
    their internal views render as an indented sub-list, but ONLY while that
    group is the current page. Clicking a sub-row writes that group's own
    focus key (`_FOCUS_KEY`) which the page reads to deep-link into exactly
    that view; clicking the group's own row clears it.

    `active` names the current page's nav key, or `None` for a legacy page
    reached only through an in-app hand-off. A click on the CURRENTLY active
    page never calls `st.switch_page` (switching to the page you are on falls
    back to the app's default page -- observed live); `st.rerun()` is what a
    same-page focus change needs."""
    registry = _nav_page_registry()

    def _go(key: str, *, sub: str | None = None) -> None:
        focus_key = _FOCUS_KEY.get(key)
        if focus_key is not None:
            if sub is not None:
                st.session_state[focus_key] = sub
            else:
                st.session_state.pop(focus_key, None)
        if key == active:
            st.rerun()
        target, url_path = registry[key]
        st.switch_page(st.Page(target, url_path=url_path))

    def _row(key: str, label: str, *, help: str | None = None) -> None:
        if _nav_row(key, label, is_active=active == key, help=help):
            _go(key)
        if active == key:
            for sub_key, sub_label in _SUBVIEWS.get(key, ()):
                if _nav_sub_row(key, sub_key, sub_label, is_active=active_sub == sub_key):
                    _go(key, sub=sub_key)

    with st.sidebar:
        st.markdown(
            '<div class="aa-nav-brand">Agentic Alpha</div>'
            '<div class="aa-nav-tagline">From news to tested research</div>',
            unsafe_allow_html=True,
        )
        for key, label in PRIMARY_NAV:
            _row(key, label)

        st.markdown('<div class="aa-nav-sep"></div><div class="aa-nav-group">Tools</div>', unsafe_allow_html=True)
        with st.container(key="navtools"):
            for key, label in TOOL_NAV:
                _row(key, label, help=_TOOL_HELP[key])

        st.markdown('<div class="aa-nav-sep"></div>', unsafe_allow_html=True)
        _row("settings", "Settings")


def render_disclaimer() -> None:
    st.caption(f"[!] {services.DISCLAIMER}")


def holdout_leak_banner(violation: str | None) -> None:
    """Call after any registry/report payload a page renders. Should never
    fire; if it ever does, fail loud in the UI rather than quietly rendering a
    locked value."""
    if violation:
        st.error(f"HOLDOUT GUARD VIOLATION -- refusing to render: {violation}")
        st.stop()
