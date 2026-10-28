"""E4 - figures: the one-dimensional example (Section 2 of the paper) and 2D shape functions.

Run from code/:  python3 -m experiments.e4_shapes
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import hbmls as hb
from experiments.style import INK, INK2, SERIES, plt, save

DIVERGING = LinearSegmentedColormap.from_list("bo", [SERIES[0], "#f0efec", SERIES[1]])


def fd2(op, y, x, h=1e-3):
    return (op([[x + h]], y)[0] - 2 * op([[x]], y)[0] + op([[x - h]], y)[0]) / h ** 2


def figure_1d():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    # (a) Hermite shape functions phi_0, psi_0 on nodes 0..3 (the 2019 notebook setting)
    nodes = [0.0, 1.0, 2.0, 3.0]
    fun = [f for p in nodes for f in hb.hermite_set([p], 1)]          # f, f' at each node
    op = hb.SingularMLS(fun, degree=2, alpha=2)
    X = np.linspace(0, 3, 601)
    phi = np.array([op.shape_functions([x]) for x in X])
    ax = axes[0]
    ax.plot(X, phi[:, 0], color=SERIES[0], label=r"$\varphi_0$ (value at 0)")
    ax.plot(X, phi[:, 1], color=SERIES[1], label=r"$\psi_0$ (slope at 0)")
    ax.plot(nodes, [0] * 4, "o", color=INK, ms=3.5, zorder=3)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_title("(a) Hermite shape functions", fontsize=9, loc="left")
    ax.set_xlabel("$x$")
    ax.legend(fontsize=7, loc="upper right", labelcolor=INK2)

    # (b) exponent rule: second derivatives reproduced only when alpha > 2
    f = lambda x: np.sin(2 * x) + 0.3 * x ** 2
    df = [f, lambda x: 2 * np.cos(2 * x) + 0.6 * x, lambda x: -4 * np.sin(2 * x) + 0.6]
    fun = [hb.derivative_1d(p, r) for p in nodes for r in (0, 1, 2)]
    y = np.array([df[g.terms[0][1][0]](g.point[0]) for g in fun])
    ax = axes[1]
    Xb = np.concatenate([np.linspace(0.9, 0.9995, 120), np.linspace(1.0005, 1.1, 120)])
    for ci, alpha in enumerate((2, 4)):
        opb = hb.SingularMLS(fun, degree=3, alpha=alpha)
        ax.plot(Xb, [fd2(opb, y, x, h=2e-4) for x in Xb], color=SERIES[ci], label=rf"$u''$, $\alpha={alpha}$")
    ax.plot([1.0], [df[2](1.0)], "o", color=INK, ms=4, zorder=3, label=r"prescribed $f''(1)$")
    ax.axvline(1.0, color=INK2, lw=0.5, ls=":")
    ax.set_title(r"(b) data $f, f', f''$: second derivative", fontsize=9, loc="left")
    ax.set_xlabel("$x$ (near the node $x=1$)")
    ax.legend(fontsize=7, loc="best", labelcolor=INK2)

    # (c) Birkhoff data: values at 0, 2 and slopes at 1, 3; plain vs completed
    g = lambda x: np.sin(1.5 * x) + 0.2 * x
    dg = lambda x: 1.5 * np.cos(1.5 * x) + 0.2
    funb = [hb.value([0.0]), hb.value([2.0]), hb.derivative_1d(1.0, 1), hb.derivative_1d(3.0, 1)]
    yb = np.array([g(0.0), g(2.0), dg(1.0), dg(3.0)])
    ax = axes[2]
    op_plain = hb.SingularMLS(funb, degree=2, alpha=2)
    fc, yc = hb.complete(funb, yb, degree=2, delta=1.5)
    op_comp = hb.SingularMLS(fc, degree=2, alpha=2)
    sl = lambda op, yy, x: (op([[x + 1e-6]], yy)[0] - op([[x - 1e-6]], yy)[0]) / 2e-6
    X3 = np.linspace(0.02, 3.0, 300)
    ax.plot(X3, [sl(op_plain, yb, x) for x in X3], color=SERIES[1], label="$u'$, Birkhoff data as given")
    ax.plot(X3, [sl(op_comp, yc, x) for x in X3], color=SERIES[0], label="$u'$, after completion")
    ax.plot([1, 3], [dg(1.0), dg(3.0)], "o", color=INK, ms=4, zorder=3, label="prescribed slopes")
    ax.set_title("(c) slopes at points without values", fontsize=9, loc="left")
    ax.set_xlabel("$x$  (values at 0, 2; slopes at 1, 3)")
    ax.legend(fontsize=6.5, loc="lower left", labelcolor=INK2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig1_one_dimensional_example")

    # numbers for the text
    print("1D example: slopes at x=1, 3  prescribed", [round(dg(1), 4), round(dg(3), 4)],
          " plain", [round(sl(op_plain, yb, p), 4) for p in (1, 3)],
          " completed", [round(sl(op_comp, yc, p), 4) for p in (1, 3)])
    for alpha in (2, 4):
        opb = hb.SingularMLS(fun, degree=3, alpha=alpha)
        print(f"alpha={alpha}: u''(1) = {fd2(opb, y, 1.0 + 1e-4):.4f} (prescribed {df[2](1.0):.4f})")


def figure_2d():
    rng = np.random.default_rng(4)
    pts = 3 * rng.random((51, 2))
    fun = [f for p in pts for f in hb.hermite_set(p, 1)]
    op = hb.SingularMLS(fun, degree=2, alpha=2, delta=0.6)
    j = int(np.argmin(np.linalg.norm(pts - 1.5, axis=1)))
    g = np.linspace(0, 3, 121)
    XX, YY = np.meshgrid(g, g)
    P = np.column_stack([XX.ravel(), YY.ravel()])
    sel = [3 * j, 3 * j + 2]                                    # value and d/dy at node j
    E = np.zeros((len(fun), 2))
    E[sel[0], 0] = 1
    E[sel[1], 1] = 1
    Z = np.array([op.coefficients(x, E)[0] for x in P])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    titles = ["Lagrange-type shape function (value datum)", r"Hermite-type shape function ($\partial_y$ datum)"]
    for ax, col, title in zip(axes, range(2), titles):
        z = Z[:, col].reshape(XX.shape)
        lim = np.max(np.abs(z))
        im = ax.pcolormesh(XX, YY, z, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        ax.plot(pts[:, 0], pts[:, 1], ".", color=INK, ms=2.5)
        ax.plot(*pts[j], "+", color=INK, ms=9, mew=1.2)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, "fig_shape_functions_2d")


if __name__ == "__main__":
    figure_1d()
    figure_2d()
