"""Shared figure style: print-friendly, recessive axes, fixed categorical order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order: blue, orange, aqua, yellow
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(f"../figures/{name}.pdf")
    fig.savefig(f"../figures/{name}.png", dpi=200)
    plt.close(fig)
