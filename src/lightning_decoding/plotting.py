from __future__ import annotations

from pathlib import Path


def save_metric_bar(summary_csv: str | Path, output_path: str | Path, metric: str) -> None:
    import pandas as pd
    from matplotlib import pyplot as plt

    data = pd.read_csv(summary_csv)
    ax = data.plot.bar(y=metric, legend=False)
    ax.set_ylabel(metric)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path)
    plt.close(ax.figure)

