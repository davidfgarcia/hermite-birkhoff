# Experiment report – step 1 (implementation and experiments), revised

*27 October 2026, revised after an independent audit (`AUDIT.md`; the corrections are
listed at the end). All numbers are reproduced by the scripts in `code/experiments/`; raw
data in `results/`, figures in `figures/`.*

## Bottom line

1. **The exactness rule (E1, E2).** Let a node xᵢ carry data q(D)f(xᵢ) for symbols q in a
   space Qᵢ, and let the singular weight be ‖x − xᵢ‖^(−α), α even. The observations support:

   > **A datum q(D)f(xᵢ) is reproduced exactly if and only if**
   > **(i)** every derivative ∂^γ q of its symbol lies in Qᵢ (closure, checked per datum), and
   > **(ii)** Δ^(α/2) q = 0 (the symbol is (α/2)-polyharmonic).

   The rule agreed with the observation for **175 of 175 data**: 92 in E1 (classical Hermite
   nodes in 1D/2D/3D with α = 2, 4, 6, and a harmonic node) and 83 in E2 (Birkhoff and
   combined functionals). No verdict changed between two stencil radii (10⁻⁶, 10⁻⁴); exact
   cases are below 10⁻²⁵ at r = 10⁻⁶, failures are at least 1.2·10⁻² (E2) and 1.7 (E1).

   Condition (ii) replaces the order condition "order < α" of the first report, which was
   wrong in 2D and 3D: in 1D it reduces to r < α, but in several variables it is the
   polyharmonic structure that matters. Examples, all observed:
   - α = 2: ∂ₓᵧ (2D) and ∂ₓᵧᵤ (3D) are exact; ∂ₓₓ, ∂ᵧᵧ, Δ and all other second/third pure or
     partly-even derivatives are not.
   - α = 2, harmonic node (symbols 1, x, y, xy, x²−y², x³−3xy², 3x²y−y³): **all exact**,
     including third-order data.
   - α = 2, {f, ∂ₓ, ∂ᵧ, ∂ₓₓ−∂ᵧᵧ}: exact; {f, ∂ₓ, ∂ᵧ, ∂ₓₓ}: ∂ₓₓ not exact.
   - α = 4 in 2D, all derivatives of order ≤ 4: everything exact except ∂ₓₓₓₓ, ∂ₓₓᵧᵧ, ∂ᵧᵧᵧᵧ
     (∂ₓₓₓᵧ and ∂ₓᵧᵧᵧ are exact: Δ² kills x³y and xy³).
   - 1D with α = 6, orders ≤ 7: orders 0–5 exact, 6 and 7 not.

2. **Komargodski–Levin §3 claim fails.** With values at some points and gradients at *other*
   points (2D), the gradients are not reproduced: errors 0.07–4.4 (E2, α = 2, degree 2,
   Gaussian decay). Re-checked in their exact setting – pure power weights ‖x − z‖^(−α), the
   minimal basis {1, x, y} and also degree 2, α = 2 and 4: errors 0.06–4.5 (24/24 gradient components; `e2c_kl_faithful.py`)
   (`results/e2c_kl_faithful.log`). With values and gradients at the same points the gradients
   are exact, as they prove. Both behaviours are predicted by condition (i).

3. **Completion repairs Birkhoff data (E2b).** Adding the missing functionals (a basis of the
   closure of the node's symbol span modulo the span) with values estimated by a smooth MLS
   fit: **19 of 19** data that failed before are exact after completion (all failing patterns
   of E2, including the three Komargodski–Levin gradient-only nodes). The 9 data at those nodes
   that were exact before stay exact. Cost: with the simple estimator used, the error of u grows
   by a factor 1.3–2.7 in the Komargodski–Levin setting – exactness is free, accuracy is not.

4. **Convergence (E3) – modest and not yet asymptotic.** At equal degree and α (m = 3, α = 2,
   4096 nodes), values + gradients beat values only by a factor 1.65 (smooth function) and 1.52
   (Franke). The larger gains in the figure (e.g. 2.8·10⁻⁸ vs 3.4·10⁻⁵) come with a higher
   degree, a different α and six times the data per node, so they are **not** a like-for-like
   comparison. Orders over the last three refinements are 2.6–4.8, but consecutive local orders
   fluctuate strongly, so **no convergence-order claim** is supported yet.

5. **Stability (E5).** The new evaluator (centred basis, log-normalised weights, SVD least
   squares, an exact two-block reformulation near a node, closed-form limit at the node) stays
   at round-off (≤ 4.7·10⁻¹⁵) for distances 10⁻¹ … 10⁻¹⁴ to a node, against a 120-digit
   reference. ε-regularised normal equations carry an O(ε) bias (≈10⁻¹⁰ for ε = 10⁻¹⁰), or, with
   ε = 10⁻¹⁶, lose up to four digits and become singular at some distances.

## Why the rule should hold (proof sketch for the paper)

Expand the MLS coefficients in the basis (y − xᵢ)^e and split a(x) − a(xᵢ) into a *free* part
F(x) (in the null space of the node functionals on P_m) and a *controlled* part C(x). Near xᵢ
the normalised weights of the other nodes are ‖x − xᵢ‖^α times smooth functions, so
C(x) = ‖x − xᵢ‖^α · G(x) with G smooth, while F is merely smooth. Then

  q(D)u(xᵢ) − q(D)p*(xᵢ) = Σ_{δ≠0} ⟨∂^δ q, F_δ⟩ + ⟨q, ‖ξ‖^α · g⟩  (Fischer pairing),

where F_δ ∈ null space and g is a polynomial built from G. The first sum vanishes for all
admissible F iff every ∂^δ q annihilates the null space, i.e. ∂^δ q ∈ Qᵢ (condition i). Since
multiplication by ‖ξ‖² is adjoint to the Laplacian under the Fischer pairing,
⟨q, ‖ξ‖^α g⟩ = c·⟨Δ^(α/2) q, g⟩, which vanishes for every g iff Δ^(α/2) q = 0 (condition ii).
The "only if" direction holds generically (for data in general position); a rigorous proof
must make precise the smoothness of G and the genericity of F and g.

## Details

### E1 – per datum (`e1_exponent_rule.py`, `results/e1_per_datum.csv`)

| case | dim | k | α | m | exact | not exact |
|---|---|---|---|---|---|---|
| Hermite | 1 | 7 | 6 | 9 | orders 0–5 | orders 6, 7 |
| Hermite | 1 | 5 | 4 | 7 | orders 0–3 | orders 4, 5 |
| Hermite | 2 | 2 | 2 | 3 | f, ∂ₓ, ∂ᵧ, ∂ₓᵧ | ∂ₓₓ, ∂ᵧᵧ |
| Hermite | 2 | 3 | 2 | 4 | f, ∂ₓ, ∂ᵧ, ∂ₓᵧ | ∂ₓₓ, ∂ᵧᵧ and all four third derivatives |
| Hermite | 2 | 3 | 4 | 4 | all 10 | – |
| Hermite | 2 | 4 | 4 | 5 | 12 incl. ∂ₓₓₓᵧ, ∂ₓᵧᵧᵧ | ∂ₓₓₓₓ, ∂ₓₓᵧᵧ, ∂ᵧᵧᵧᵧ |
| harmonic node | 2 | 3 | 2 | 4 | all 7 | – |
| Hermite | 3 | 2 | 2 | 3 | f, ∂ₓ, ∂ᵧ, ∂ᵤ, ∂ₓᵧ, ∂ₓᵤ, ∂ᵧᵤ | ∂ₓₓ, ∂ᵧᵧ, ∂ᵤᵤ |
| Hermite | 3 | 3 | 2 | 4 | the above + ∂ₓᵧᵤ | the other 9 third derivatives, ∂ₓₓ, ∂ᵧᵧ, ∂ᵤᵤ |

### E2 – Birkhoff patterns (`e2_birkhoff.py`, `results/e2_birkhoff.csv`)

83 data, 21 not exact, all as predicted. Not exact: every datum above a gap in 1D; ∂ₓ alone;
∂ₓₓ in {f, ∂ₓₓ}; ∂ₓᵧ in {f, ∂ₓ, ∂ₓᵧ}; Δ in {f, Δ}; gradient-only nodes; at α = 2: ∂ₓₓ in
{f, ∂ₓ, ∂ᵧ, ∂ₓₓ} and Δ in {f, ∂ₓ, ∂ᵧ, Δ}. Exact, among others: {f, ∂ₓ, ∂ᵧ, Δ} at α = 6,
{f, D_v}, {f, ∂ₓ, ∂ᵧ, ∂ₓᵧ} and {f, ∂ₓ, ∂ᵧ, ∂ₓₓ−∂ᵧᵧ} at α = 2.

### E2b – completion (`e2b_completion.py`)

(a) 28 data at patched nodes: 19 failing before → 19 exact after; 9 exact before → exact after.
(b) Values at n Halton points, gradients at n other Halton points, Franke, m = 3, α = 2:

| n | plain: max error | plain: gradient mismatch | completed: max error | completed: mismatch (FD step 10⁻⁵ / 10⁻⁶) |
|---|---|---|---|---|
| 64 | 4.5e-2 | 3.6e-1 | 5.9e-2 | 1.9e-7 / 2.0e-9 |
| 256 | 1.7e-2 | 1.7e-1 | 3.1e-2 | 1.4e-6 / 1.4e-8 |
| 1024 | 1.2e-3 | 4.4e-2 | 3.4e-3 | 1.2e-6 / 1.2e-8 |

The completed mismatch falls by 100× when the step falls by 10×: it is the O(step²) error of
the finite-difference check; the method's own mismatch is zero (as E2b(a) shows exactly).

### E3 – convergence (`e3_convergence.py`, `plot_e3.py`, figure `e3_convergence`)

Max error on [0.1, 0.9]²; δ = 2h is the Gaussian decay length, h = N^(−1/2) (not the fill
distance of the Halton set).

| data | m | α | smooth f, 4096 nodes | order (last 3) | Franke, 4096 nodes | order |
|---|---|---|---|---|---|---|
| values | 2 | 2 | 3.4e-5 | 2.6 | 6.4e-4 | 2.7 |
| values | 3 | 2 | 3.5e-6 | 3.6 | 3.7e-4 | 3.3 |
| values + gradients | 3 | 2 | 2.1e-6 | 4.0 | 2.4e-4 | 3.4 |
| values + gradients + Hessians | 4 | 4 | 2.8e-8 | 4.1 | 1.1e-5 | 4.8 |

### E5 – stability (`e5_stability.py`, figure `e5_stability`)

Worst error vs a 120-digit reference along x = 2 + t, t = 10⁻¹ … 10⁻¹⁴: stable evaluator
4.4e-16 (k = 1) and 4.7e-15 (k = 2); normal equations ε = 10⁻¹⁰: 9.1e-11 and 1.6e-10;
ε = 10⁻¹⁶: 6.1e-15 and 3.8e-12, singular at t = 10⁻¹¹, 10⁻¹⁴ (k = 1) and 10⁻⁶ (k = 2).

### E6 – degenerate configurations and error identities (`e6_degenerate.py`)

Condition (i) fails, yet the datum is reproduced (Theorem 4.8 / Prop. 4.11 of the paper);
a small perturbation then breaks it:

| configuration | reproduced? | perturbed |
|---|---|---|
| 1D f'(0), f(1), f'(5/2), m=1, α=2 (interpolatory restriction) | yes (<1e-36) | + f(4): errors 0.21–2.1 |
| 2D K=2, gradient-only node, α=2 (single scale) | yes | + third node: 0.03–0.46 |
| 2D K=2, {f, Dxx} node, α=4 | yes | + third node: 2.9–3.3 |
| 2D Dx at 0, other nodes on the x2-axis (symmetric) | yes | moved 0.05: 0.08–0.16 |

Error identities at a Hermite k=3 node, α=2 (signed, 12 digits): e(Dxx)=e(Dyy),
e(Dxxx)=3e(Dxyy), e(Dyyy)=3e(Dxxy).

## Open questions

- **Proof** of the rule, both directions; in particular the genericity argument for "only if".
- **Convergence**: a parameter study (decay length δ/h, α, degree, node sets that reach the
  boundary) before the paper states any order.
- **Completion**: a better estimator for the missing data (larger support, higher degree, or a
  global solve) should remove the accuracy penalty.
- **Geometry**: every exactness test uses one random node set per case; exactness is linear in
  the data, but the node geometry, α and m should be varied once more for the paper.

## Corrections made after the audit

| audit finding | correction |
|---|---|
| "iff r < α, no exception" false in 2D/3D (mixed derivatives exact at α = 2) | rule replaced by (i) + (ii); E1 re-run per datum, 92/92 |
| r ≥ α never tested for α ≥ 6 | 1D k = 7, α = 6 added: orders 6, 7 fail as predicted |
| "17 of 17 previously failing" | E2b re-run with before/after bookkeeping: 19 failing → 19 exact |
| KL gradient-only errors 0.07–5.1 | 0.07–4.4 (5.1 belonged to another pattern) |
| factor 1.4–2.7 | 1.3–2.7 |
| Franke values m = 3: 3.6e-4 | 3.7e-4 |
| "errors are larger near the edges" | removed (false for Franke in 19 of 28 runs) |
| Hermite vs Lagrange gain confounded | like-for-like factor (1.65, 1.52) stated; figure legend now gives α and data per node |
| no precision guard in the stencil check | precision now raised automatically; every verdict checked at two radii; errors kept in high precision |
| E5 label "100-digit"; zeros drawn as spikes; breakdowns unexplained | label 120-digit; zeros and singular cases marked and explained |
| KL counterexample used this code's weights | re-checked with their exact weights and minimal basis |
