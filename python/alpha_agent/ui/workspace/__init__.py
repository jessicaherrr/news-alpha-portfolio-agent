"""The Research Thread workspace: shell (stepper, context panel, navigation)
and one renderer per step. Every module here RENDERS canonical backend
objects resolved by `alpha_agent.ui.research_thread.ThreadPipeline`; none of
them scores, ranks, sizes, validates or recalls anything itself (statically
guarded by `tests/python/test_research_thread_workspace.py`)."""
