"""
src/llm_integration/__init__.py
===================================
Module d'intégration LLM pour la génération d'explications de fraudes.
"""

from .llm_helper import LLMHelper, FEATURE_LABELS, FEATURE_INTERPRETATIONS

__all__ = ["LLMHelper", "FEATURE_LABELS", "FEATURE_INTERPRETATIONS"]
