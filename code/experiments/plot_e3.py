"""Re-draw the E3 convergence figure from results/e3_convergence.csv."""
import csv

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from experiments.e3_convergence import CONFIGS, FUNCS
from experiments.style import INK2, SERIES, plt, save

rows = list(csv.DictReader(open("../results/e3_convergence.csv")))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax, fn in zip(axes, FUNCS):
    for ci, (label, k, m, alpha) in enumerate(CONFIGS):
        rs = [r for r in rows if int(r["k"]) == k and int(r["degree"]) == m and r["function"] == fn]
        hs = np.array([float(r["h"]) for r in rs])
        es = np.array([float(r["max_err"]) for r in rs])
        ax.loglog(hs, es, "o-", color=SERIES[ci], ms=3, label=label)
        rate = np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]
        ax.annotate(f"{rate:.1f}", (hs[-1], es[-1]), xytext=(4, [3, 3, -4, 0][ci]), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.invert_xaxis()
    ticks = [1 / 8, 1 / 16, 1 / 32, 1 / 64]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1/8", "1/16", "1/32", "1/64"])
    ax.set_xlim(0.14, 0.0125)
    ax.set_xlabel(r"$h = N_{\mathrm{nodes}}^{-1/2}$")
    ax.set_title("smooth test function" if fn == "smooth" else "Franke function", fontsize=9)
axes[0].set_ylabel(r"max error on $[0.1,0.9]^2$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.1, 1, 1))
save(fig, "e3_convergence")
