"""E5 - evaluation near a node: epsilon-regularised normal equations (the usual practice,
and the 2019 notebooks) versus the stable evaluator, against a 120-digit reference.

Run from code/:  python3 -m experiments.e5_stability
"""
import csv

import numpy as np

import hbmls as hb
from experiments.style import INK2, SERIES, plt, save


def naive(fun, y, x, degree, alpha, eps):
    """Normal equations, raw monomials, weights 1/(|x - p|^alpha + eps)."""
    exps = hb.multi_indices(1, degree)
    P = np.array([f.basis_row(exps, np.zeros(1), 1.0) for f in fun])      # monomials in x
    w = np.array([1.0 / (abs(x - f.point[0]) ** alpha + eps) for f in fun])
    S = P.T @ (w[:, None] * P)
    a = np.linalg.solve(S, P.T @ (w * y))
    return float(np.array([x ** e[0] for e in exps]) @ a)


if __name__ == "__main__":
    nodes = [1.0, 2.0, 3.0, 4.0]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (k, alpha, m) in zip(axes, [(1, 2, 3), (2, 4, 4)]):
        fun = [f for p in nodes for f in hb.hermite_set([p], k)]
        y = np.random.default_rng(3).normal(size=len(fun))
        op = hb.SingularMLS(fun, degree=m, alpha=alpha)
        ref = hb.MPSingularMLS(op, dps=120)
        ts = np.logspace(-1, -14, 27)
        series = {"stable evaluator": [], r"normal equations, $\varepsilon=10^{-10}$": [],
                  r"normal equations, $\varepsilon=10^{-16}$": []}
        for t in ts:
            x = 2.0 + t
            r = float(ref.value([x], y))
            series["stable evaluator"].append(abs(op([[x]], y)[0] - r))
            for eps, key in ((1e-10, r"normal equations, $\varepsilon=10^{-10}$"),
                             (1e-16, r"normal equations, $\varepsilon=10^{-16}$")):
                try:
                    series[key].append(abs(naive(fun, y, x, m, alpha, eps) - r))
                except np.linalg.LinAlgError:
                    series[key].append(np.nan)
            rows.append(dict(k=k, alpha=alpha, t=t, **{kk: v[-1] for kk, v in series.items()}))
        FLOOR = 1e-17
        for ci, (key, vals) in enumerate(series.items()):
            v = np.array(vals, dtype=float)
            shown = np.where(v > 0, v, np.nan)
            ax.loglog(ts, np.maximum(shown, FLOOR), "o-" if ci == 0 else ".-", ms=2.5,
                      color=SERIES[ci], label=key)
            z = v == 0                                    # exact agreement: open marker on the floor
            ax.loglog(ts[z], np.full(z.sum(), FLOOR), "o", mfc="none", ms=3.5, color=SERIES[ci])
            bad = np.isnan(v)                             # singular normal equations
            ax.loglog(ts[bad], np.full(bad.sum(), 3e-9), "x", ms=5, mew=1.2, color=SERIES[ci])
        ax.set_ylim(5e-18, 1e-8)
        ax.invert_xaxis()
        ax.set_xlabel("distance to the node $x_i = 2$")
        ax.set_title(rf"$k={k}$, $\alpha={alpha}$, degree $m={m}$", fontsize=9)
        print(f"k={k} alpha={alpha}: worst error  " +
              "  ".join(f"{key.split(',')[0] if ',' not in key else key}: {np.nanmax(v):.1e}" for key, v in series.items()))
    axes[0].set_ylabel("error vs 120-digit reference")
    axes[1].annotate("x: singular matrix\no: exactly zero error", (0.98, 0.5), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=6.5, color=INK2)
    axes[0].legend(fontsize=6.5, loc="upper left", labelcolor=INK2)
    save(fig, "e5_stability")
    with open("../results/e5_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
