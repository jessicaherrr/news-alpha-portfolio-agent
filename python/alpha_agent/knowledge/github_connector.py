"""Alpha Discovery live-research campaign, Checkpoint 5 -- the REAL, read-only
GitHub strategy-research connector.

SAFETY (unchanged from `alpha_agent.knowledge.external_sources`'s boundary):
this module only ever calls the GitHub REST API for TEXT and METADATA --
repository search, repo metadata, README, license, default-branch commit SHA,
and a small, bounded set of plain-text source files. It NEVER clones a
repository, NEVER imports or executes anything it retrieves, and NEVER shells
out. A repository's code is DATA to read, not something this platform runs.

AUTHENTICATION: reads `GITHUB_TOKEN` from the process environment (optionally
loaded from a local, gitignored `.env` via `python-dotenv` --
`connector_support.load_dotenv_if_available`). The token is NEVER logged,
printed, persisted, or included in any cached/committed artifact -- only a
boolean "was a token used" and the resulting `ConnectorHealth` are ever
surfaced.

BUDGETS: bounded by `ConnectorBudget` (task spec "NETWORK BUDGET" /
"GITHUB SEARCH QUERY GENERATION") -- at most `max_queries` searches, at most
`max_search_results` repositories considered per query, at most
`max_files_per_repo` source files fetched per repository, and the connector
stops once it has produced `max_documents_total` knowledge items across the
whole call.
"""
from __future__ import annotations

import base64
import os
from enum import Enum

from pydantic import BaseModel

from alpha_agent.knowledge.connector_support import (
    ConnectorBudget,
    ConnectorHttpError,
    classify_mechanism,
    http_get_json,
    load_dotenv_if_available,
    now_iso,
    provenance_hash,
    truncate,
)
from alpha_agent.knowledge.models import (
    IngestionResult,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)

__all__ = [
    "GITHUB_API_VERSION",
    "GITHUB_CANDIDATE_FILE_PATTERNS",
    "GitHubConnectorHealth",
    "GitHubStrategyConnector",
    "generate_github_queries",
    "github_health",
]

GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

#: Task spec "GITHUB SOURCE SELECTION" -- only these path SHAPES are ever
#: considered candidates for a per-file fetch beyond the README. Deliberately
#: excludes vendor/lockfile/binary/generated paths.
GITHUB_CANDIDATE_FILE_PATTERNS: tuple[str, ...] = (
    "strategy.py", "signal.py", "signals.py", "indicators.py", "indicator.py",
    "backtest.py", "backtester.py", "config.py", "config.yaml", "config.yml",
)
_CANDIDATE_DOC_PREFIX = "docs/"
_MAX_FILE_BYTES = 60_000

#: Generic, non-market-specific futures/CTA research terms (task spec section
#: 20) -- always included alongside market-specific queries, bounded by
#: `ConnectorBudget.max_queries`.
_GENERIC_QUERY_TERMS: tuple[str, ...] = (
    "futures trading strategy", "CTA strategy trend following",
    "systematic futures strategy", "managed futures strategy",
)


class GitHubConnectorHealth(str, Enum):
    CONNECTED = "CONNECTED"
    NOT_CONNECTED = "NOT_CONNECTED"
    RATE_LIMITED = "RATE_LIMITED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class GitHubHealthReport(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    health: GitHubConnectorHealth
    token_present: bool
    detail: str = ""
    #: `None` unless a real, authenticated `/rate_limit` call succeeded.
    remaining_requests: int | None = None
    limit_requests: int | None = None


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_health(*, token: str | None = None, timeout: float = 10.0) -> GitHubHealthReport:
    """Real, read-only health probe (`GET /rate_limit`). Never raises -- every
    outcome is a typed `GitHubHealthReport`. The token value itself is never
    included in the report."""
    load_dotenv_if_available()
    tok = token if token is not None else os.environ.get("GITHUB_TOKEN")
    if not tok:
        return GitHubHealthReport(
            health=GitHubConnectorHealth.NOT_CONNECTED, token_present=False,
            detail="GITHUB_TOKEN is not set -- no live GitHub call attempted.",
        )
    try:
        body, _headers_out = http_get_json(
            f"{GITHUB_API}/rate_limit", headers=_headers(tok), timeout=timeout,
        )
    except ConnectorHttpError as exc:
        if exc.status_code == 401:
            return GitHubHealthReport(
                health=GitHubConnectorHealth.ERROR, token_present=True,
                detail="GitHub rejected the configured token (401 Unauthorized).",
            )
        return GitHubHealthReport(
            health=GitHubConnectorHealth.ERROR, token_present=True, detail=f"GitHub health check failed: {exc}",
        )
    core = (body or {}).get("resources", {}).get("core", {})
    remaining = core.get("remaining")
    limit = core.get("limit")
    if isinstance(remaining, int) and remaining <= 0:
        return GitHubHealthReport(
            health=GitHubConnectorHealth.RATE_LIMITED, token_present=True,
            detail="GitHub API rate limit exhausted.", remaining_requests=remaining, limit_requests=limit,
        )
    return GitHubHealthReport(
        health=GitHubConnectorHealth.CONNECTED, token_present=True,
        detail="Authenticated GitHub API access confirmed.", remaining_requests=remaining, limit_requests=limit,
    )


def generate_github_queries(
    *, market: str, objective: str = "", failure_notes: tuple[str, ...] = (), budget: ConnectorBudget | None = None,
) -> tuple[str, ...]:
    """A bounded, deterministic set of GitHub search queries (task spec
    "GITHUB SEARCH QUERY GENERATION"). Market-specific terms first, then
    generic CTA/futures terms, truncated to `budget.max_queries`."""
    budget = budget or ConnectorBudget()
    market_name = {
        "NQ": "Nasdaq futures", "ES": "S&P 500 futures", "CL": "crude oil futures",
        "GC": "gold futures", "ZN": "Treasury futures",
    }.get(market, f"{market} futures")
    specific = (
        f"{market_name} trading strategy",
        f"{market_name} breakout strategy",
        f"{market_name} mean reversion",
        f"{market_name} volatility trend",
    )
    queries = list(dict.fromkeys((*specific, *_GENERIC_QUERY_TERMS)))
    return tuple(queries[: budget.max_queries])


def _score_repo(repo: dict, *, query: str) -> float:
    """Deterministic quality/relevance score (task spec "GITHUB REPOSITORY
    QUALITY FILTER" -- explicitly NOT primarily stars). Stars enter as one
    small, capped term only."""
    score = 0.0
    description = (repo.get("description") or "").lower()
    query_terms = [t for t in query.lower().split() if len(t) > 3]
    score += 10.0 * sum(1 for t in query_terms if t in description)
    if repo.get("license"):
        score += 15.0
    if not repo.get("archived"):
        score += 5.0
    if not repo.get("fork"):
        score += 5.0
    size = repo.get("size") or 0  # KB -- a signal the repo has real content, not a stub
    if size > 50:
        score += 10.0
    stars = repo.get("stargazers_count") or 0
    score += min(stars, 200) / 40.0  # capped, minor tie-break only
    return score


def _decode_readme(body: dict) -> str:
    content = body.get("content", "")
    encoding = body.get("encoding", "base64")
    if encoding != "base64":
        return content
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- a malformed README must not crash ingestion
        return ""


class GitHubStrategyConnector:
    """Implements `alpha_agent.knowledge.external_sources.ExternalSourceAdapter`.
    Real, read-only, budget-bounded. Falls back honestly (never fabricates a
    result) when no token is configured, the token is rejected, or the rate
    limit is exhausted."""

    def __init__(self, *, token: str | None = None, budget: ConnectorBudget | None = None, timeout: float | None = None):
        load_dotenv_if_available()
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._budget = budget or ConnectorBudget()
        self._timeout = timeout if timeout is not None else self._budget.timeout_seconds

    @property
    def source_type(self) -> SourceType:
        return SourceType.GITHUB

    def health(self) -> GitHubHealthReport:
        return github_health(token=self._token, timeout=self._timeout)

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        report = self.health()
        if report.health is GitHubConnectorHealth.NOT_CONNECTED:
            return IngestionResult(source_type=SourceType.GITHUB, status=IngestionStatus.NOT_CONNECTED, items=(), detail=report.detail)
        if report.health is GitHubConnectorHealth.RATE_LIMITED:
            return IngestionResult(source_type=SourceType.GITHUB, status=IngestionStatus.RATE_LIMITED, items=(), detail=report.detail)
        if report.health is GitHubConnectorHealth.ERROR:
            return IngestionResult(source_type=SourceType.GITHUB, status=IngestionStatus.ERROR, items=(), detail=report.detail)

        market = markets[0] if markets else ""
        queries = generate_github_queries(market=market or "futures", objective=query, budget=self._budget) if not query else (query,)
        headers = _headers(self._token)

        items: list[StrategyKnowledgeItem] = []
        degraded_notes: list[str] = []
        seen_repos: set[str] = set()

        for q in queries:
            if len(items) >= self._budget.max_documents_total:
                break
            try:
                body, _h = http_get_json(
                    f"{GITHUB_API}/search/repositories",
                    headers=headers, timeout=self._timeout,
                    params={"q": q, "sort": "updated", "order": "desc", "per_page": self._budget.max_search_results},
                )
            except ConnectorHttpError as exc:
                degraded_notes.append(f"search {q!r} failed: {exc}")
                continue
            repos = body.get("items", []) or []
            ranked = sorted(repos, key=lambda r: _score_repo(r, query=q), reverse=True)
            for repo in ranked:
                if len(items) >= self._budget.max_documents_total:
                    break
                full_name = repo.get("full_name")
                if not full_name or full_name in seen_repos:
                    continue
                seen_repos.add(full_name)
                item = self._normalize_repo(repo, headers=headers, degraded_notes=degraded_notes)
                if item is not None:
                    items.append(item)

        status = IngestionStatus.CONNECTED if not degraded_notes else IngestionStatus.DEGRADED
        detail = (
            f"{len(items)} repositor{'y' if len(items)==1 else 'ies'} normalized into knowledge items across "
            f"{len(queries)} quer{'y' if len(queries)==1 else 'ies'}."
        )
        if degraded_notes:
            detail += " Some fetches failed: " + "; ".join(degraded_notes[:3])
        return IngestionResult(source_type=SourceType.GITHUB, status=status, items=tuple(items), detail=detail)

    def _fetch_candidate_files(
        self, full_name: str, default_branch: str, headers: dict[str, str], degraded_notes: list[str],
    ) -> tuple[str, tuple[str, ...]]:
        """Task spec "GITHUB SOURCE SELECTION": walk the repository tree once,
        select only the small set of PATH SHAPES that plausibly hold strategy
        logic or documentation, and fetch at most `max_files_per_repo` of
        them as TEXT (never vendor code, lockfiles, binaries, or generated
        artifacts)."""
        try:
            tree_body, _h = http_get_json(
                f"{GITHUB_API}/repos/{full_name}/git/trees/{default_branch}",
                headers=headers, timeout=self._timeout, params={"recursive": "1"},
            )
        except ConnectorHttpError as exc:
            degraded_notes.append(f"{full_name} tree unavailable: {exc}")
            return "", ()

        candidate_paths: list[str] = []
        for entry in tree_body.get("tree", []) or []:
            if entry.get("type") != "blob":
                continue
            path = entry.get("path", "")
            lower = path.lower()
            name = lower.rsplit("/", 1)[-1]
            is_doc = lower.startswith(_CANDIDATE_DOC_PREFIX) and lower.endswith(".md")
            if name not in GITHUB_CANDIDATE_FILE_PATTERNS and not is_doc:
                continue
            size = entry.get("size") or 0
            if size and size > _MAX_FILE_BYTES:
                continue
            candidate_paths.append(path)
            if len(candidate_paths) >= self._budget.max_files_per_repo:
                break

        snippets: list[str] = []
        fetched: list[str] = []
        for path in candidate_paths:
            try:
                file_body, _h = http_get_json(
                    f"{GITHUB_API}/repos/{full_name}/contents/{path}", headers=headers, timeout=self._timeout,
                )
            except ConnectorHttpError as exc:
                degraded_notes.append(f"{full_name}/{path} fetch failed: {exc}")
                continue
            text = _decode_readme(file_body)  # identical base64-text decode
            if text:
                snippets.append(f"--- {path} ---\n{truncate(text, 800)}")
                fetched.append(path)
        return "\n\n".join(snippets), tuple(fetched)

    def _normalize_repo(self, repo: dict, *, headers: dict[str, str], degraded_notes: list[str]) -> StrategyKnowledgeItem | None:
        full_name = repo["full_name"]
        default_branch = repo.get("default_branch", "main")
        description = repo.get("description") or ""

        readme_text = ""
        try:
            readme_body, _h = http_get_json(
                f"{GITHUB_API}/repos/{full_name}/readme", headers=headers, timeout=self._timeout,
            )
            readme_text = _decode_readme(readme_body)
        except ConnectorHttpError as exc:
            degraded_notes.append(f"{full_name} README unavailable: {exc}")

        combined_text = f"{repo.get('name', '')} {description}\n{readme_text}"
        mechanism = classify_mechanism(combined_text)
        if mechanism is None:
            return None  # honest skip -- never a guessed mechanism (task spec section 8/36)

        file_snippets, fetched_paths = self._fetch_candidate_files(
            full_name, default_branch, headers, degraded_notes
        )

        commit_sha: str | None = None
        try:
            commit_body, _h = http_get_json(
                f"{GITHUB_API}/repos/{full_name}/commits/{default_branch}",
                headers=headers, timeout=self._timeout,
            )
            commit_sha = commit_body.get("sha")
        except ConnectorHttpError as exc:
            degraded_notes.append(f"{full_name} default-branch commit lookup failed: {exc}")

        license_name = None
        lic = repo.get("license") or {}
        if isinstance(lic, dict):
            license_name = lic.get("spdx_id") or lic.get("name")

        quality = SourceQualityTier.TIER_B if (license_name and len(readme_text) > 200) else SourceQualityTier.TIER_C

        provenance = provenance_hash(
            "github1",
            {"repo": full_name, "commit_sha": commit_sha, "description": description, "files": fetched_paths},
        )
        notes = truncate(readme_text, 1200)
        if file_snippets:
            notes = notes + "\n\n" + truncate(file_snippets, 2000)
        # Release UX bugfix pass: a real repository's own file paths are
        # arbitrary, repo-owner-controlled strings that can incidentally
        # contain a date-shaped substring (e.g. a filename like
        # "2026-07-15-plan.md") with no bearing on market data at all -- but
        # `limitations` (unlike `implementation_notes`) reaches
        # `StrategyKnowledgeItem.render_snippet()` and therefore
        # `ResearchContext`/`OrchestratorConfig.knowledge_base`, which the
        # locked-holdout guard scans for a >= 2025-01-01 date. A real GitHub
        # repo's plan/roadmap files dated in 2025/2026 (increasingly common
        # now that real-world content IS dated 2026) would otherwise make
        # "Connected Research" crash on a false positive. State a COUNT here,
        # never the literal paths -- `implementation_notes` (never scanned by
        # the holdout guard) still carries the real per-file snippets for
        # anyone inspecting Research Inspiration detail.
        implementation_risk = (
            f"{len(fetched_paths)} additional source file(s) inspected as TEXT ONLY (never imported/executed)."
            if fetched_paths else "No additional source files selected beyond the README."
        )
        return StrategyKnowledgeItem(
            knowledge_id=provenance,
            source_type=SourceType.GITHUB,
            source_quality=quality,
            ingestion_status=IngestionStatus.CONNECTED,
            title=repo.get("full_name", full_name),
            authors=(repo.get("owner", {}).get("login", ""),) if repo.get("owner") else (),
            source_url=repo.get("html_url"),
            repository=full_name,
            commit_sha=commit_sha,
            license=license_name,
            accessed_at=now_iso(),
            publication_date=repo.get("created_at"),
            markets=(),
            asset_classes=("futures",),
            economic_mechanism=mechanism,
            entry_logic_summary=truncate(description or "See README.", 300),
            implementation_notes=notes,
            reported_results=None,  # NEVER our evidence -- a GitHub README's claims stay external
            limitations=(
                "External GitHub source; mechanism independently re-implemented, never copied verbatim. "
                + implementation_risk
            ),
            provenance_hash=provenance,
        )
