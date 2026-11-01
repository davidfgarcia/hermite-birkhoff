"""Third derivatives of SWIMLS2D3: agreement with SWIMLS2D, exactness on polynomials, and
consistency with a central difference of the exact Hessian rows."""
import numpy as np
import pytest

from hbmls.swimls2d import SWIMLS2D
from hbmls.swimls2d3 import SWIMLS2D3


def _nodes(h=0.1, jit=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(-1, 1 + 1e-9, h)
    X, Y = np.meshgrid(t, t)
    P = np.c_[X.ravel(), Y.ravel()]
    return P + jit * h * rng.uniform(-1, 1, P.shape)


POLY = {"u": lambda x, y: x**5 - 2 * x**2 * y**3 + y**4 * x + 3 * x * y,
        "dx": lambda x, y: 5 * x**4 - 4 * x * y**3 + y**4 + 3 * y,
        "dy": lambda x, y: -6 * x**2 * y**2 + 4 * y**3 * x + 3 * x,
        "lap": lambda x, y: 20 * x**3 - 4 * y**3 - 12 * x**2 * y + 12 * y**2 * x}
THIRD = lambda x, y: np.array([60 * x**2, -12 * y**2, -24 * x * y + 12 * y**2, -12 * x**2 + 24 * x * y])


@pytest.mark.parametrize("syms,alpha", [(("u", "dx", "dy"), 2), (("u", "dx", "dy", "lap"), 4)])
def test_third_derivatives(syms, alpha):
    P = _nodes()
    op = SWIMLS2D3(P, syms, 6, alpha, 0.1, 1e-24)
    base = SWIMLS2D(P, syms, 6, alpha, 0.1, 1e-24)
    data = np.array([POLY[s](x, y) for x, y in P for s in syms])
    x0 = np.array([0.0123, -0.0377])
    u, g, H, T = op.phi3(x0)
    b0, b1, b2 = base.phi(x0)
    assert np.allclose(u, b0, atol=1e-13) and np.allclose(g, b1, atol=1e-12)
    assert np.allclose(H, b2, atol=1e-10)
    assert np.max(np.abs(T @ data - THIRD(*x0))) < 1e-9          # degree-5 data reproduced
    eps = 1e-4                                                   # FD of the exact Hessian rows
    Hx = (op.phi3(x0 + [eps, 0])[2] - op.phi3(x0 - [eps, 0])[2]) / (2 * eps)
    Hy = (op.phi3(x0 + [0, eps])[2] - op.phi3(x0 - [0, eps])[2]) / (2 * eps)
    smooth = np.array([[np.sin(2 * x + y), 2 * np.cos(2 * x + y), np.cos(2 * x + y),
                        -5 * np.sin(2 * x + y)][: len(syms)] for x, y in P]).ravel()
    fd = np.array([Hx[0] @ smooth, Hx[1] @ smooth, Hx[2] @ smooth, Hy[2] @ smooth])
    assert np.max(np.abs(T @ smooth - fd)) < 1e-6 * max(1.0, np.max(np.abs(fd)))
