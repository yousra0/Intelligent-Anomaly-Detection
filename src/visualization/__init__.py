"""Visualization package exports.

Some legacy preparation plotting functions may not be present depending on
the current project state, so imports are optional to avoid package-level
ImportError when loading submodules.
"""

# Baseline model plotting helpers used by 03_baseline_models.ipynb
from .model_plots import (
    plot_confusion_matrices,
    plot_feature_importances,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_analysis,
)

# Optional legacy preparation plotting helpers.
try:
    from .prep_plots import (
        plot_all_preparation_figures,
        plot_log_transform,
        plot_scaling_effect,
        plot_smote_effect,
        plot_split,
    )
except ImportError:
    pass

__all__ = [
    "plot_pr_curves",
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_feature_importances",
    "plot_threshold_analysis",
    "plot_model_comparison",
]

for _name in [
    "plot_log_transform",
    "plot_split",
    "plot_scaling_effect",
    "plot_smote_effect",
    "plot_all_preparation_figures",
]:
    if _name in globals():
        __all__.append(_name)
