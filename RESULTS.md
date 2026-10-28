# Experiment report – step 1 (implementation and experiments)

*27 October 2026. All numbers below are reproduced by the scripts in `code/experiments/`;
raw data in `results/`, figures in `figures/`.*

## Bottom line

1. **Exponent rule (E1).** With classical Hermite data at a node (all derivatives up to
   order k) and singular weights ‖x − xᵢ‖^(−α), α even, the derivative of order r is
   reproduced exactly **if and only if r < α**. Observed in 1D (k ≤ 5, α ≤ 8), 2D (k ≤ 3,
   α ≤ 6) and 3D (k ≤ 2, α ≤ 4) with no exception. This is sharper than the Komargodski–Levin
   conjecture (α = 2k): α = 4 already reproduces third derivatives.
2. **D-invariance rule (E2).** At a node, a datum q(D)f(xᵢ) is reproduced exactly iff every
   derivative of its symbol q lies in the span of that node's symbols (the node's data are
   *D-invariant*), given order < α. The prediction agreed with the observation for
   **67 of 67** data, including the delicate cases: {f, Δf} fails but {f, ∂ₓf, ∂ᵧf, Δf} is
   exact; a directional derivative with the value is exact; ∂ₓᵧ with only f, ∂ₓ fails.
3. **Komargodski–Levin §3 claim fails (E2).** With values at some points and gradients at
   *other* points (2D, α = 2), the gradients are **not** reproduced (errors 0.07–5.1 at the
   three gradient-only nodes checked); with values and gradients at the same points they are.
4. **Completion repairs Birkhoff data (E2b).** Adding the missing lower-order functionals
   (closure of the node's symbol span), with values estimated by a smooth MLS fit, makes
   **17 of 17** previously failing data exact. Caveat: with the simple estimator used, the
   function error grows by a factor 1.4–2.7 (table below) – exactness is free, accuracy is not.
5. **Convergence (E3).** Hermite data are much more accurate than Lagrange data at the same
   nodes (smooth function, 4096 nodes: 3.4·10⁻⁵ values-only m = 2 → 2.8·10⁻⁸ with
   gradients + Hessians, m = 4). Observed orders are close to but not cleanly m + 1
   (2.6–4.8 over the last three refinements); not yet asymptotic, see open questions.
6. **Stability (E5).** The new evaluator (centred basis, log-normalised weights, SVD least
   squares, two-block solve near a node, closed-form limit at the node) stays at round-off
   (≤ 5·10⁻¹⁵) at all distances 10⁻¹…10⁻¹⁴ from a node. ε-regularised normal equations, as
   in the 2019 notebooks, carry an O(ε) bias (≈10⁻¹⁰ for ε = 10⁻¹⁰) or lose up to four digits
   and occasionally break down (ε = 10⁻¹⁶).

## Details

### E1 – exponent rule (`e1_exponent_rule.py`, `results/e1_*.csv`)

Error |Dʳu(xᵢ) − yᵢ| at the nodes (high-precision finite-difference stencils, 60 digits);
"0" means below 10⁻¹⁵, failures are O(1)–O(10³).

| dim | Hermite order k | α = 2 | α = 4 | α = 6 | α = 8 |
|---|---|---|---|---|---|
| 1 | 1 | exact | exact | exact | exact |
| 1 | 2 | order 2 fails | exact | exact | exact |
| 1 | 3 | orders 2–3 fail | exact | exact | exact |
| 1 | 4 | orders 2–4 fail | order 4 fails | exact | exact |
| 1 | 5 | orders 2–5 fail | orders 4–5 fail | exact | exact |
| 2 | 1 / 2 / 3 | k = 1 exact; order ≥ 2 fails | exact | exact | – |
| 3 | 1 / 2 | k = 1 exact; order 2 fails | exact | – | – |

### E2 – Birkhoff patterns (`e2_birkhoff.py`, `results/e2_birkhoff.csv`)

Exact: {f, f′, f″, f‴}, {f, f′, f″}, f alone below a gap (e.g. f in {f, f″});
{f, ∂ₓ}, {f, ∂ₓ, ∂ᵧ}, {f, D_v}, {f, ∂ₓ, ∂ᵧ, ∂ₓᵧ}, {f, ∂ₓ, ∂ᵧ, Δ}, full Hermite k = 2.
Not exact (as predicted): every datum above a gap in 1D ({f″} in {f, f″}, {f′} alone,
{f″} alone, {f′, f″}, {f‴} in {f, f′, f‴}, …); ∂ₓ alone; ∂ₓₓ in {f, ∂ₓₓ}; ∂ₓᵧ in {f, ∂ₓ, ∂ₓᵧ};
Δ in {f, Δ}; gradient-only nodes (Komargodski–Levin setting).

### E2b – completion (`e2b_completion.py`)

Values at n Halton points, gradients at n other Halton points, Franke function, m = 3, α = 2:

| n | plain: max error | plain: gradient mismatch | completed: max error | completed: gradient mismatch |
|---|---|---|---|---|
| 64 | 4.5e-2 | 2.7e-1 | 5.9e-2 | 8.6e-10 |
| 256 | 1.7e-2 | 7.3e-2 | 3.1e-2 | 9.5e-10 |
| 1024 | 1.2e-3 | 3.0e-2 | 3.4e-3 | 1.0e-9 |

(the 10⁻⁹ level is the central-difference noise of the check, not a method error)

### E3 – convergence (`e3_convergence.py`, figure `e3_convergence`)

Max error on [0.1, 0.9]² (Halton nodes do not reach the boundary; near the edges the method
extrapolates and errors are larger – recorded in the CSV). Support δ = 2h, derivative rows
scaled by h^order.

| data | m | α | smooth f, 4096 nodes | order (last 3) | Franke, 4096 nodes | order |
|---|---|---|---|---|---|---|
| values | 2 | 2 | 3.4e-5 | 2.6 | 6.4e-4 | 2.7 |
| values | 3 | 2 | 3.5e-6 | 3.6 | 3.6e-4 | 3.3 |
| values + gradients | 3 | 2 | 2.1e-6 | 4.0 | 2.4e-4 | 3.4 |
| values + gradients + Hessians | 4 | 4 | 2.8e-8 | 4.1 | 1.1e-5 | 4.8 |

### E5 – stability (`e5_stability.py`, figure `e5_stability`)

Worst error against a 120-digit reference along x = 2 + t, t = 10⁻¹ … 10⁻¹⁴:
stable evaluator 4.4e-16 (k = 1) and 4.7e-15 (k = 2); normal equations ε = 10⁻¹⁰: 9.1e-11 and
1.6e-10; ε = 10⁻¹⁶: 6.1e-15 and 3.8e-12 with breakdowns (singular matrix) at some t.

## What this means for the main theorem

The data support the following statement, which the proof should establish:

> Let the data at node xᵢ be q(D)f(xᵢ), q ∈ Qᵢ, with Qᵢ a space of polynomial symbols of
> degree ≤ kᵢ ≤ m, and let w(x, xᵢ) ~ ‖x − xᵢ‖^(−α) with α even. If Qᵢ is D-invariant and
> kᵢ < α, SWIMLS reproduces all data at xᵢ exactly. Conversely, a datum whose symbol has a
> derivative outside Qᵢ is in general not reproduced.

Proof sketch (to be made rigorous): near xᵢ the dominant weight forces the components of the
local coefficient vector a(x) controlled by Qᵢ to equal their node values up to O(‖x − xᵢ‖^α),
so their derivatives of order < α vanish at xᵢ; by the Leibniz rule, q(D)u(xᵢ) = q(D)p*(xᵢ) +
Σ_{γ≠0} (terms pairing D^γ a(xᵢ) with the derivatives ∂^γ q of the symbol), and those terms vanish
exactly when every ∂^γ q lies in Qᵢ.

## Open questions

- Convergence orders are not yet cleanly m + 1. Candidates: support size δ relative to h,
  the decay factor, pre-asymptotic behaviour. Needs a parameter study before the paper
  makes any rate claim.
- Completion: a better estimator for the missing data (larger support, higher degree,
  or a global solve) should remove the accuracy penalty.
- Exponent rule: rigorous proof that r < α is also *necessary* in general (the data show
  failure for every r ≥ α tested).
