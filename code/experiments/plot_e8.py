"""Figure for E8: max error of u and of grad u vs h, compactly supported weights."""
import csv

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from experiments.style import INK2, SERIES, plt, save

rows = list(csv.DictReader(open("../results/e8_compact.csv")))
SEL = [(0, 3, 2, 3.0), (1, 3, 2, 2.0), (1, 4, 2, 2.0), (2, 5, 4, 2.0)]
NAMES = {0: "values", 1: "values + gradients", 2: "values + gradients + Hessians"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ci, (k, m, a, c) in enumerate(SEL):
    rs = [r for r in rows if (int(r["k"]), int(r["degree"]), int(r["alpha"]), float(r["c"])) == (k, m, a, c)]
    h = np.array([float(r["h"]) for r in rs])
    for ax, key, order in ((axes[0], "max_err", m + 1), (axes[1], "grad_max_err", m)):
        e = np.array([float(r[key]) for r in rs])
        ax.loglog(h, e, "o-", ms=3, color=SERIES[ci], label=f"{NAMES[k]}, $m={m}$" if ax is axes[0] else None)
        ref = e[-1] * (h / h[-1]) ** order                     # reference slope through the last point
        ax.loglog(h[-4:], ref[-4:], ":", color=SERIES[ci], lw=1)
for ax, t in zip(axes, ["(a) $\\max|u-f|$", "(b) $\\max|\\nabla(u-f)|$"]):
    ax.xaxis.set_major_locator(FixedLocator([1 / 8, 1 / 16, 1 / 32, 1 / 64]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1/8", "1/16", "1/32", "1/64"])
    ax.set_xlim(0.14, 0.009)
    ax.set_xlabel(r"$h=N^{-1/2}$")
    ax.set_title(t, fontsize=9, loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=2, labelcolor=INK2,
           bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, "e8_compact")
