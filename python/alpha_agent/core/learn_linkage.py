"""Phase 5 -- Learn linkage. ``alpha_agent.learn`` already has no Futures
hardcoding anywhere in its content (``concepts.py``, ``strategy_explainers.py``,
``pipeline_explainer.py``, ``autopsy.py`` all read generic, mechanism/registry
-driven state) -- Phase 5 does not need to change it, only confirm and name
the linkage surface a future generation's UI would deep-link into: a
``Concept`` looked up by its stable ``concept_id`` string, exactly how
``alpha_agent.ui.learn_links.render_learn_why`` already links to it today.
"""
from __future__ import annotations

from alpha_agent.learn.concepts import Concept, get_concept

__all__ = ["Concept", "get_concept"]
