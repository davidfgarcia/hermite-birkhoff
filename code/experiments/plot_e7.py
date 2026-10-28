"""Figure for E7: max error vs h and vs number of data (delta = 2h)."""
import csv

import numpy as np

from matplotlib.ticker import FixedLocator, NullLocator

from experiments.style import INK2, SERIES, plt, save

rows = list(csv.DictReader(open("../results/e7_parameters.csv"))) + \
    list(csv.DictReader(open("../results/e7b_values_high_degree.csv")))
DATA = {0: 1, 1: 3, 2: 6}
NAMES = {0: "values", 1: "values + gradients", 2: "values + gradients + Hessians"}
SEL = [(0, 3, 2), (1, 3, 2), (2, 3, 4), (0, 5, 2), (2, 5, 4)]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for k, m, a in SEL:
    rs = [r for r in rows if (int(r["k"]), int(r["degree"]), int(r["alpha"]), float(r["c"])) == (k, m, a, 2.0)]
    n = np.array([int(r["nodes"]) for r in rs])
    e = np.array([float(r["max_err"]) for r in rs])
    style = dict(color=SERIES[k], ls="-" if m == 3 else "--", marker="o", ms=3)
    lab = f"{NAMES[k]}, $m={m}$"
    axes[0].loglog(1 / np.sqrt(n), e, label=lab, **style)
    axes[1].loglog(n * DATA[k], e, **style)
for ax in axes:
    ax.set_ylabel("max error on $[0.1,0.9]^2$") if ax is axes[0] else None
axes[0].xaxis.set_major_locator(FixedLocator([1 / 8, 1 / 16, 1 / 32, 1 / 64]))
axes[0].xaxis.set_minor_locator(NullLocator())
axes[0].set_xticklabels(["1/8", "1/16", "1/32", "1/64"])
axes[0].set_xlim(0.14, 0.0125)
axes[0].set_xlabel(r"$h=N_{\mathrm{nodes}}^{-1/2}$")
axes[1].set_xlabel("number of data")
axes[0].set_title("(a) per node spacing", fontsize=9, loc="left")
axes[1].set_title("(b) per number of data", fontsize=9, loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=3, labelcolor=INK2, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "e7_convergence")
