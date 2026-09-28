"""Mission Part E / task spec section 55 -- static proof that the
market-OBSERVATION plane (`alpha_agent.marketdata.databento_provider` /
`alpha_agent.ui.databento_context`) can never reach the SCIENTIFIC plane
(2018-2022 Fast Screen, 2023-2024 strict validation, Research Promise, the
locked 2025 holdout), and that the scientific plane can never reach INTO the
observation provider either. Both directions matter: a one-way "observation
reads a static, pure derivation helper from `alpha_agent.data.
contract_economics`" is the one sanctioned exception (documented in
`databento_provider`'s own module docstring) and is asserted explicitly
below, never left to a blanket "no cross-import" rule that would also forbid
that one safe reuse.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules that are the OBSERVATION plane -- current/delayed market data for
#: display and conversational context only.
OBSERVATION_MODULES = (
    "alpha_agent.marketdata.databento_provider",
    "alpha_agent.marketdata.databento_schemas",
    "alpha_agent.marketdata.databento_config",
    "alpha_agent.ui.databento_context",
    # Market Intelligence + Futures Universe campaign, Checkpoint A: the
    # static product catalog and typed capability schema are pure
    # marketdata-plane data/shape modules -- no network, no registry, and
    # (per this test) never a scientific-plane import either. The dynamic
    # composition of catalog + observation + read-only registry evidence
    # lives in `alpha_agent.ui.market_universe` instead, which is a normal
    # UI module (like `alpha_agent.ui.dashboard`), deliberately NOT listed
    # here -- see that module's own docstring for why it is allowed to read
    # the registry.
    "alpha_agent.marketdata.product_catalog",
    "alpha_agent.marketdata.capability",
    # Checkpoint C/D (Market Intelligence Data Completion Pass): the
    # contract-ladder/term-structure schemas and the related-market
    # registry + cross-market metrics are the same kind of pure,
    # network-free marketdata-plane shape/logic module as the two above.
    "alpha_agent.marketdata.relative_markets",
)

#: Package prefixes that make up the SCIENTIFIC plane -- research corpus
#: acquisition, discovery/generation, fast screen, and validation/registry.
SCIENTIFIC_PACKAGE_PREFIXES = (
    "alpha_agent.data",
    "alpha_agent.discovery",
    "alpha_agent.screening",
    "alpha_agent.validation",
    "alpha_agent.registry",
    "alpha_agent.strategy",
    "alpha_agent.holdout",
)

#: The one sanctioned one-way exception: a pure, stateless, never-hardcoded
#: economics-derivation function -- not the acquisition/registry pipeline.
_ALLOWED_OBSERVATION_IMPORTS_INTO_SCIENTIFIC = frozenset({"alpha_agent.data.contract_economics"})


def _module_path(module_name: str) -> Path:
    return REPO_ROOT / "python" / Path(*module_name.split(".")).with_suffix(".py")


def _module_imports(module_name: str) -> set[str]:
    """Pure static analysis -- reads the file's own source text, never
    imports/executes the module. Robust to an empty `__init__.py` (no
    source lines to speak of -> no imports)."""
    path = _module_path(module_name)
    source = path.read_text(encoding="utf-8") if path.exists() else ""
    if not source.strip():
        return set()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _all_python_modules_under(package_prefix: str) -> list[str]:
    pkg_path = REPO_ROOT / "python" / package_prefix.replace(".", "/")
    if not pkg_path.is_dir():
        return []
    out = []
    for path in sorted(pkg_path.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT / "python").with_suffix("")
        out.append(".".join(rel.parts))
    return out


@pytest.mark.parametrize("module_name", OBSERVATION_MODULES)
def test_observation_module_only_reaches_scientific_plane_via_sanctioned_exception(module_name):
    imports = _module_imports(module_name)
    for imported in imports:
        if not any(imported == p or imported.startswith(p + ".") for p in SCIENTIFIC_PACKAGE_PREFIXES):
            continue
        assert imported in _ALLOWED_OBSERVATION_IMPORTS_INTO_SCIENTIFIC, (
            f"{module_name} imports {imported!r}, which is on the SCIENTIFIC plane "
            "and not the one sanctioned exception -- observation-plane code must "
            "never depend on the research acquisition/discovery/validation/registry pipeline"
        )


@pytest.mark.parametrize("package_prefix", SCIENTIFIC_PACKAGE_PREFIXES)
def test_scientific_plane_never_imports_the_observation_provider(package_prefix):
    for module_name in _all_python_modules_under(package_prefix):
        imports = _module_imports(module_name)
        for imported in imports:
            assert not (imported.startswith("alpha_agent.marketdata.databento") or imported == "alpha_agent.ui.databento_context"), (
                f"{module_name} (scientific plane, {package_prefix}) imports {imported!r} -- "
                "the observation provider must never feed the research pipeline"
            )


def test_observation_provider_never_writes_to_raw_or_processed_data_paths():
    """Checks actual code identifiers (the raw-store API this codebase's
    scientific pipeline writes through), not prose -- this module's own
    docstring legitimately mentions "data/raw/" while explaining why it never
    touches it."""
    path = _module_path("alpha_agent.marketdata.databento_provider")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_names = {"RAW_ROOT", "store_raw", "RawArtifact", "fetch_and_store_raw"}
    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used_names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (used_names & forbidden_names), used_names & forbidden_names


def test_observation_provider_never_touches_the_experiment_registry():
    for module_name in OBSERVATION_MODULES:
        imports = _module_imports(module_name)
        assert not any("registry" in i for i in imports), (
            f"{module_name} must never import a registry module"
        )
