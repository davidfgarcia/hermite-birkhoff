"""E3 - h-convergence on scattered (Halton) nodes in 2D for the Franke function.

Compares Lagrange data (values only) with Hermite data (values + gradients, and
values + gradients + Hessians) at the same nodes.  Errors are measured on a uniform
41 x 41 grid of [0,1]^2.

Run from code/:  python3 -m experiments.e3_convergence
"""
import csv
import time
from multiprocessing import Pool

import numpy as np
import sympy as sp

import hbmls as hb
from experiments.style import INK2, SERIES, plt, save

x_, y_ = sp.symbols("x y")
FRANKE = (sp.Rational(3, 4) * sp.exp(-((9 * x_ - 2) ** 2 + (9 * y_ - 2) ** 2) / 4)
          + sp.Rational(3, 4) * sp.exp(-(9 * x_ + 1) ** 2 / 49 - (9 * y_ + 1) / 10)
          + sp.Rational(1, 2) * sp.exp(-((9 * x_ - 7) ** 2 + (9 * y_ - 3) ** 2) / 4)
          - sp.Rational(1, 5) * sp.exp(-(9 * x_ - 4) ** 2 - (9 * y_ - 7) ** 2))
_DER = {}


def deriv(beta, p):
    if beta not in _DER:
        _DER[beta] = sp.lambdify((x_, y_), sp.diff(FRANKE, x_, beta[0], y_, beta[1]), "numpy")
    return float(_DER[beta](*p))


def halton(n, dim, skip=31):
    primes = [2, 3, 5][:dim]
    out = np.zeros((n, dim))
    for j, b in enumerate(primes):
        for i in range(n):
            k, f, r = i + skip, 1.0, 0.0
            while k > 0:
                f /= b
                r += f * (k % b)
                k //= b
            out[i, j] = r
    return out


# label, Hermite order k, local degree m, alpha
CONFIGS = [
    ("values, m = 2", 0, 2, 2),
    ("values, m = 3", 0, 3, 2),
    ("values + gradients, m = 3", 1, 3, 2),
    ("values + gradients + Hessians, m = 4", 2, 4, 4),
]
NS = [64, 128, 256, 512, 1024, 2048]
G = np.linspace(0, 1, 41)
GRIDPTS = np.array([(a, b) for a in G for b in G])
EXACT = np.array([deriv((0, 0), p) for p in GRIDPTS])


def run(args):
    label, k, m, alpha, n = args
    pts = halton(n, 2)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    h = 1 / np.sqrt(n)
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=2.0 * h, length=h, cutoff=1e-14)
    yv = op.data_from(deriv)
    u = op(GRIDPTS, yv)
    e = u - EXACT
    return dict(config=label, k=k, degree=m, alpha=alpha, nodes=n, data=len(fun), h=h,
                max_err=float(np.max(np.abs(e))), rms_err=float(np.sqrt(np.mean(e ** 2))))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(c[0], c[1], c[2], c[3], n) for c in CONFIGS for n in NS]
    with Pool(2) as pool:
        rows = pool.map(run, jobs)
    with open("../results/e3_convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ci, (label, k, m, alpha) in enumerate(CONFIGS):
        rs = [r for r in rows if r["config"] == label]
        hs = np.array([r["h"] for r in rs])
        es = np.array([r["max_err"] for r in rs])
        ds = np.array([r["data"] for r in rs])
        rate = np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]
        print(f"{label:<40} max errors {np.array2string(es, precision=1)}   rate h^{rate:.2f}")
        for ax, xs in zip(axes, (hs, ds)):
            ax.loglog(xs, es, "o-", color=SERIES[ci], ms=3.5, label=f"{label}  (rate {rate:.1f})")
    axes[0].set_xlabel("fill distance $h \\approx N_{\\mathrm{nodes}}^{-1/2}$")
    axes[0].invert_xaxis()
    axes[1].set_xlabel("number of data (all functionals)")
    axes[0].set_ylabel("max error on $[0,1]^2$")
    axes[0].legend(loc="lower left", fontsize=7, labelcolor=INK2)
    save(fig, "e3_convergence")
    print(f"{time.time() - t0:.0f}s")
