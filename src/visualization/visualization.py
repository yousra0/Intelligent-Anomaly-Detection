### Distribution des types de transaction
import matplotlib.pyplot as plt
from pathlib import Path


def plot_transaction_types(
    df,
    figsize=(10, 5),
    colors=None,
    title="Distribution des types de transaction",
    save_path: str | Path | None = None,
):
    """Plot transaction type distribution with optional save to disk."""
    counts = df["type"].value_counts()

    if colors is None:
        colors = ["#1976D2", "#E53935", "#43A047", "#FB8C00", "#6D4C41"]

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", color=colors[: len(counts)], ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Type", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig