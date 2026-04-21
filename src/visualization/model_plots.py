"""Baseline model plotting helpers.

This module re-exports plotting functions currently implemented in
`src.visualization.prep_plots` to keep notebook imports stable.
"""

try:
    from .prep_plots import (
        plot_confusion_matrices,
        plot_feature_importances,
        plot_model_comparison,
        plot_pr_curves,
        plot_roc_curves,
        plot_threshold_analysis,
    )
except ImportError:
    from src.visualization.prep_plots import (
        plot_confusion_matrices,
        plot_feature_importances,
        plot_model_comparison,
        plot_pr_curves,
        plot_roc_curves,
        plot_threshold_analysis,
    )

__all__ = [
    "plot_pr_curves",
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_feature_importances",
    "plot_threshold_analysis",
    "plot_model_comparison",
]
