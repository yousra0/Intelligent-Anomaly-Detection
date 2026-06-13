"""
app/services/llm_service.py
Wrapper autour de LLMHelper avec chargement depuis llm_config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def get_llm_helper(project_root: Path):
    """Instancie LLMHelper depuis la config YAML du projet."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.llm_integration.llm_helper import LLMHelper

    config_path = project_root / "config" / "llm_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen = cfg.get("generation", {})
    return LLMHelper(
        config_path=str(config_path),
        timeout=gen.get("timeout", 30),
        temperature=gen.get("temperature", 0.1),
        max_tokens=gen.get("max_tokens", 400),
    )
