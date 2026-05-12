"""
src/ollama_integration/__init__.py
===================================
Module d'intégration Ollama pour la génération d'explications de fraudes.
"""

from .ollama_helper import OllamaHelper, FEATURE_LABELS, FEATURE_INTERPRETATIONS

__all__ = ["OllamaHelper", "FEATURE_LABELS", "FEATURE_INTERPRETATIONS"]

