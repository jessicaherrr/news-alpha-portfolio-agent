"""Tests for `alpha_agent.knowledge.provider_registry` (Release UX Part K,
task spec section 42).
"""
from __future__ import annotations

from alpha_agent.knowledge.practitioner_connector import INSTITUTION_ALLOWLIST
from alpha_agent.knowledge.provider_registry import (
    PRACTITIONER_PROVIDERS,
    ProviderCategory,
    name_fragments_allowlist,
    provider_for_fragment,
    providers_by_category,
)


def test_registry_is_broad_not_a_single_institution():
    assert len(PRACTITIONER_PROVIDERS) >= 10
    assert len({p.category for p in PRACTITIONER_PROVIDERS}) >= 3


def test_every_provider_has_a_category_domain_and_organization():
    for provider in PRACTITIONER_PROVIDERS:
        assert provider.organization
        assert provider.category in ProviderCategory
        assert provider.name_fragments


def test_connector_allowlist_is_derived_from_the_registry_not_a_second_list():
    assert set(INSTITUTION_ALLOWLIST) == set(name_fragments_allowlist())


def test_providers_by_category_filters_correctly():
    central_banks = providers_by_category(ProviderCategory.CENTRAL_BANK)
    assert all(p.category == ProviderCategory.CENTRAL_BANK for p in central_banks)
    assert any("federal reserve" in p.name_fragments for p in central_banks)


def test_provider_for_fragment_resolves_organization():
    provider = provider_for_fragment("federal reserve")
    assert provider is not None
    assert provider.organization == "Federal Reserve System"


def test_provider_for_fragment_unknown_returns_none():
    assert provider_for_fragment("not a real institution") is None


def test_no_provider_is_uniquely_authoritative_by_construction():
    """Structural: the registry is a tuple of many, never a single entry a
    caller could mistake for 'the' authoritative source."""
    organizations = {p.organization for p in PRACTITIONER_PROVIDERS}
    assert len(organizations) == len(PRACTITIONER_PROVIDERS)  # no duplicates either
    assert len(organizations) > 1
