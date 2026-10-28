"""E3 - h-convergence on scattered (Halton) nodes in 2D.

Compares Lagrange data (values only) with Hermite data (values + gradients, and
values + gradients + Hessians) at the same nodes, for a smooth test function and for
Franke's function.  Errors are measured on a uniform 33 x 33 grid of the interior
square [0.1, 0.9]^2 (Halton points do not reach the boundary, so near the edges the
method extrapolates; the boundary error is reported separately in the CSV).

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
SMOOTH = sp.exp(x_ + y_ / 2) * sp.cos(2 * y_) + sp.sin(3 * x_ * y_)
FUNCS = {"smooth": SMOOTH, "franke": FRANKE}
_DER = {}


def deriv(beta, p, name="franke"):
    key = (name, tuple(beta))
    if key not in _DER:
        _DER[key] = sp.lambdify((x_, y_), sp.diff(FUNCS[name], x_, beta[0], y_, beta[1]), "numpy")
    return float(_DER[key](*p))


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
    ("values (1 datum/node), m = 2, α = 2", 0, 2, 2),
    ("values (1 datum/node), m = 3, α = 2", 0, 3, 2),
    ("values + gradients (3/node), m = 3, α = 2", 1, 3, 2),
    ("values + gradients + Hessians (6/node), m = 4, α = 4", 2, 4, 4),
]
NS = [64, 128, 256, 512, 1024, 2048, 4096]
G = np.linspace(0.1, 0.9, 33)
GRIDPTS = np.array([(a, b) for a in G for b in G])
EXACT = np.array([deriv((0, 0), p) for p in GRIDPTS])            # franke (used by e2b)
GB = np.linspace(0, 1, 33)
BND = np.array([(a, b) for a in GB for b in GB if min(a, b, 1 - a, 1 - b) < 0.1])


def run(args):
    fname, label, k, m, alpha, n = args
    pts = halton(n, 2)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    h = 1 / np.sqrt(n)
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=2.0 * h, length=h, cutoff=1e-14)
    yv = op.data_from(lambda b, p: deriv(b, p, fname))
    e = op(GRIDPTS, yv) - np.array([deriv((0, 0), p, fname) for p in GRIDPTS])
    eb = op(BND, yv) - np.array([deriv((0, 0), p, fname) for p in BND])
    return dict(function=fname, config=label, k=k, degree=m, alpha=alpha, nodes=n, data=len(fun), h=h,
                max_err=float(np.max(np.abs(e))), rms_err=float(np.sqrt(np.mean(e ** 2))),
                boundary_max_err=float(np.max(np.abs(eb))))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(fn, c[0], c[1], c[2], c[3], n) for fn in FUNCS for c in CONFIGS for n in NS]
    with Pool(2) as pool:
        rows = pool.map(run, jobs)
    with open("../results/e3_convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
    for ax, fn in zip(axes, FUNCS):
        for ci, (label, k, m, alpha) in enumerate(CONFIGS):
            rs = [r for r in rows if r["config"] == label and r["function"] == fn]
            hs = np.array([r["h"] for r in rs])
            es = np.array([r["max_err"] for r in rs])
            rate = np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]
            print(f"{fn:<7} {label:<40} max err {np.array2string(es, precision=1)}  rate(last 3) h^{rate:.2f}")
            ax.loglog(hs, es, "o-", color=SERIES[ci], ms=3, label=f"{label} (slope {rate:.1f})")
        ax.invert_xaxis()
        ax.set_xlabel("$h = N_{\\mathrm{nodes}}^{-1/2}$")
        ax.set_title("smooth test function" if fn == "smooth" else "Franke function", fontsize=9)
        ax.legend(loc="lower left", fontsize=6.5, labelcolor=INK2)
    axes[0].set_ylabel("max error on $[0.1,0.9]^2$")
    save(fig, "e3_convergence")
    print(f"{time.time() - t0:.0f}s")
