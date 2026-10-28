# Audit of RESULTS.md (27 Oct 2026)

Scope: RESULTS.md was checked against `results/*.csv|log`, the code in `code/`, and the figures. Independent spot checks were run from a scratch directory. No project file was changed apart from this one.

## 1. Tests

`python3 -m tests.test_core` passes all **15 of 15** checks (P_m dimensions, polynomial reproduction on and off the nodes, interpolation, continuity, float-vs-mpmath agreement, stencil exactness on P_3, and float = 100-digit reference near gapped nodes).

## 2. Number-by-number check

Rates are least-squares slopes of log(max_err) against log(h) over the last three h values, recomputed from `e3_convergence.csv`.

| # | Claim (location) | Report | Data | Status |
|---|---|---|---|---|
| 1 | E1 1D table, every cell (k=1..5, α=2..8) | as tabulated | e1_1d.csv agrees cell by cell | OK |
| 2 | E1 failures are O(1)–O(10³) | O(1)–O(10³) | 4.0 … 1643 | OK |
| 3 | E1 2D, α=2: "order ≥ 2 fails" | all of order 2 fails | **∂xy is exact** at α=2 (k=2 and k=3). Only ∂xx and ∂yy fail (185 / 69.5). The CSV keeps only the maximum per order, which hides this | **MISMATCH** |
| 4 | E1 3D, α=2: "order 2 fails" | all of order 2 fails | **∂xy, ∂xz, ∂yz are exact**. Only ∂xx, ∂yy, ∂zz fail (34.5) | **MISMATCH** |
| 5 | Exponent rule "**iff r < α** … with no exception" | iff | The "if" direction holds in every case tested. The "only if" direction fails for mixed second derivatives at α=2 in 2D and 3D | **OVERCLAIM / FALSE** |
| 6 | Open questions: "failure for every r ≥ α tested" | – | False, for the same mixed-derivative cases. Also, r ≥ α was only tested for α=2 and α=4. No r ≥ 6 was ever run | **MISMATCH** |
| 7 | α=4 reproduces 3rd derivatives | yes | 1D k=3 and 2D k=3, α=4: all exact | OK |
| 8 | E2 prediction agreement | 67 of 67 | 67/67 (log and CSV) | OK as a count, but see §3.5 |
| 9 | E2 exact / not-exact pattern lists (Details) | – | match the log | OK |
| 10 | KL gradient-only node errors | 0.07–**5.1** | 0.073–**4.40** (5.1 is the ∂xy error from pattern {f,∂x,∂xy}) | **MISMATCH** |
| 11 | KL: same points means gradients reproduced | yes | nodes 0–2 exact | OK |
| 12 | E2b "**17 of 17 previously failing** data exact" | 17 previously failing | Only **10** of the 17 were failing in E2. The other 7 (f and ∂x data at the patched nodes) were already exact. Nine E2 failures were never re-tested in E2b(a): {0,3}∂³, {0,2,3}∂²/∂³, and the 6 KL gradient-only data | **OVERCLAIM** (see spot check S3: those 9 do become exact) |
| 13 | Completion error-growth factor | 1.4–2.7 | 1.33, 2.01, 1.89, 1.78, 2.71, so **1.3–2.7** | **MISMATCH (minor)** |
| 14 | E2b table, n = 64 / 256 / 1024 (all 12 numbers) | – | match (1.65e-2 rounds to 1.7e-2, 3.35e-3 rounds to 3.4e-3) | OK |
| 15 | "10⁻⁹ level is central-difference noise" | assertion | Plausible (step 1e-6, about 1e-10 round-off times scaling). Not demonstrated: the step was not varied, and only `gp[:20]` was checked, not all n points | UNVERIFIED |
| 16 | E3 smooth 4096: values m=2 / m=3 / +grad / +Hess | 3.4e-5 / 3.5e-6 / 2.1e-6 / 2.8e-8 | 3.37e-5 / 3.47e-6 / 2.10e-6 / 2.81e-8 | OK |
| 17 | E3 Franke 4096 | 6.4e-4 / **3.6e-4** / 2.4e-4 / 1.1e-5 | 6.37e-4 / **3.67e-4** / 2.42e-4 / 1.07e-5 | 3.6e-4 should be **3.7e-4** (minor) |
| 18 | E3 rates, smooth (last 3) | 2.6 / 3.6 / 4.0 / 4.1 | 2.56 / 3.62 / 4.04 / 4.08 | OK |
| 19 | E3 rates, Franke (last 3) | 2.7 / 3.3 / 3.4 / 4.8 | 2.67 / 3.29 / 3.42 / 4.75 | OK |
| 20 | Rate range | 2.6–4.8 | 2.56–4.75 | OK |
| 21 | "Hermite data are **much more accurate** than Lagrange at the same nodes" (3.4e-5 → 2.8e-8) | ×1200 | The comparison mixes m=2, α=2 with m=4, α=4, and 1× with 6× the data count. At equal m=3, α=2, values+gradients beats values by only **×1.65** (smooth) and **×1.52** (Franke). There is no values-only m=4 run | **OVERCLAIM** |
| 22 | "near the edges … errors are larger" | always larger | True for smooth f. For **Franke, boundary error < interior error in 19 of 28 runs** | **MISMATCH** |
| 23 | E5 stable evaluator worst error | 4.4e-16 / 4.7e-15; ≤ 5e-15 | 4.44e-16 / 4.66e-15 | OK |
| 24 | E5 ε=1e-10 | 9.1e-11 / 1.6e-10 | 9.14e-11 / 1.61e-10 | OK |
| 25 | E5 ε=1e-16 | 6.1e-15 / 3.8e-12, breakdowns at some t | 6.05e-15 / 3.81e-12. NaN at t=1e-11 and 1e-14 (k=1) and at t=1e-6 (k=2) | OK |
| 26 | E5 reference precision | 120 digits | dps=120 in code. But the figure y-label and the script docstring say "100-digit" | text OK, **figure MISMATCH** |
| 27 | "as in the 2019 notebooks" | – | notebooks not in repo | UNVERIFIABLE |

## 3. Methodology concerns (most severe first)

1. **Results are aggregated per order, which hid a counterexample to the headline rule (high).** In `e1_exponent_rule.py`, `worst[f.order] = max(...)` collapses every multi-index of one order into a single number. Re-running per component (S2 below) shows that at α=2 the mixed second derivatives are reproduced exactly, while ∂xx, ∂yy (and ∂zz) fail with identical errors. A 60-digit stencil check shows the stencil Hessian equals the data Hessian plus c·I, with c = 185.13 in 2D k=2. Directional second differences in 5 directions, at r=1e-5 and 1e-7, give exactly vᵀ(H_data + cI)v. So u is genuinely C² at the node, with a term c‖x−xᵢ‖²: the failure is real, not a stencil artefact. The rule should be stated as "D^β fails iff D^β applied to ‖x−xᵢ‖^α (or higher-order analogues) is non-zero". At least, it is not simply "r < α". The "iff" claim and the theorem's "Conversely" direction need to be revised.
2. **Very small samples behind "no exception" and "67 of 67" (medium).**
   - 2D/3D E1: one random configuration (seed 3 or 5), one checked node, one random data vector.
   - E2: one configuration per pattern, at a single α (8 in 1D, 6 in 2D) and a single degree.
   - α ≥ 6 was never tested with r ≥ α.
   - Of the 67 data, 48 are exact and many of those are trivial (values are always interpolated).

   Exactness is linear in y, so one generic y is fine. Geometry, α and m were not varied, though.
3. **E3 comparison is confounded (medium).** Hermite and Lagrange runs differ in m, α and number of data. The rates fluctuate strongly between consecutive refinements:
   - smooth +Hessians: 2.6, 4.4, 6.2, 5.2, 4.5, 3.7
   - Franke +grad: 0.03 … 4.0
   - RMS-based last-3 rates are 2.9–5.0, versus 2.6–4.8 for max-error

   "Order (last 3)" is therefore not robust and should not appear in a paper without the parameter study. Also, δ is a Gaussian decay length, not a "support", and h = N^(-1/2) is not the fill distance of the Halton set.
4. **EXACT = 1e-9 and the float cast (low; conclusions unaffected).** `check_node` rounds the 60-digit result to double before subtracting y, so it cannot resolve errors below about 1e-16·|y|. Any error in (1e-16, 1e-9) would be labelled "exact". In practice every "exact" entry is 0.0 (one is 2.8e-17) and the smallest failure is 1.2e-2, so the classification has a gap of more than 7 decades. Recommendation: keep the error in mp, and show that it scales like r^(extra+1) as r decreases (S1 does this).
5. **Precision margin of the stencils (low).** At dps=60 and r=1e-6 the 5th-derivative stencil loses about 30 digits, which is fine (S1). With dps=30 or 40 the same exact case gives **false failures**: ∂⁵ error 192 or 768, ∂⁴ error 1e-4. Any future run with higher order or smaller r needs dps ≳ 20 + order·|log₁₀ r|. Nothing in the code enforces this.
6. **Minor issues.**
   - The KL "§3 claim fails" result uses this code's formulation (α=2, m=2, δ=1). It should be confirmed that this matches KL's exact weights and degree before claiming their result fails.
   - E2b(b) gradient mismatch is checked at only 20 points.
   - README refers to a `paper/` directory that does not exist.
   - `common.not_d_invariant` is unused.

### Spot checks (independent; errors kept in mp, varying r and dps)

- **S1 (stability of exact vs failing).**
  - 1D k=4, α=4: orders 0–3 give errors of 1e-22 to 1e-63 that scale like r^6 (stencil truncation), at every (dps, r) in {(60,1e-6), (60,1e-4), (100,1e-8)}. ∂⁴ = 101.0 in all cases.
  - 1D k=5, α=8: all orders are exact at those three settings. The low-dps false failures are listed in §3.5.
  - 2D {f,∂x,∂xy}: ∂xy = 5.09 in all five settings. f and ∂x are ≤ 1e-21.
  - 2D {f,Δ}: Δ = 6.85 in all settings.

  The exact / not-exact verdicts are stable.
- **S2 (per-component E1).**
  - 2D k=2 α=2: ∂xx = ∂yy = 185, **∂xy exact** at all 5 (dps, r) settings.
  - 2D k=3 α=2: ∂xy exact, all third derivatives fail.
  - 3D k=2 α=2: **∂xy, ∂xz, ∂yz exact**.
- **S3 (completion of the untested E2 failures).** {0,3} (adds ∂, ∂²), {0,2,3} (adds ∂), and the KL gradient-only nodes 8–10 (α=2) are all exact after completion. The claim "completion repairs Birkhoff data" therefore extends to all 19 failures.

## 4. Prediction logic review

- **`predicted()` (e2_birkhoff.py).** It differentiates the symbol repeatedly, one variable at a time, and checks each derivative against span(S) with a relative tolerance of 1e-9. It returns False at the first derivative outside the span, and it pushes the raw derivative (not its projection), which is correct. The loop terminates because the degree drops at each step. Revisits are redundant but harmless.

  It encodes the per-datum criterion "every ∂^γ q ∈ Q_i". That is the criterion actually tested, but it is weaker than the node-level "the node's data are D-invariant" wording in Bottom line §2. For example, f in {f, f″} is exact even though the node is not D-invariant. The wording should match the criterion.

  It ignores the order/α condition. That is harmless in E2 because the order is always below α by design. It does mean the combined rule was never tested where both conditions interact; the mixed-derivative effect in §3.1 is exactly such a case.
- **`missing_functionals()` (birkhoff.py).** It closes an orthonormal basis of the symbol span under ∂/∂ξ_j. Every basis vector and every new residual is differentiated, so the span of everything differentiated equals the closure, and by linearity this is sufficient. It then returns an orthonormal basis of closure ⊖ span. This is correct, and it agrees with the "added" columns in the log ({1}→f; {2}→∂, f; {0,2}→∂; Laplacian→∂x, ∂y). The sign of the added functional (e.g. "-1Dx") is arbitrary and cosmetic.
- **`complete()`.** Missing values come from a Gaussian-weighted LS fit, and its basis scaling matches `basis_row(…, delta)`. Exactness of the original data does not depend on these values, which is consistent with the data.

## 5. Figures

- **e5_stability.png.**
  - The y-label says "error vs **100-digit** reference", but the code uses 120 digits.
  - Exact zeros are clipped to 1e-18 and drawn as deep downward spikes. These look like real data points; they should be marked or the y-range floored at about 1e-17.
  - Breakdowns (NaN) appear only as unexplained gaps in the ε=1e-16 curve. They should be annotated.
- **e3_convergence.png.**
  - The legend omits α: the m=4 curve uses α=4, the others α=2. It also omits the data counts (1×, 3×, 6×), so the visual "Hermite is better" comparison is confounded (§3.3).
  - The slope labels 3.6 and 4.0 overlap in the smooth panel.
  - Slopes are last-3 fits and hide erratic local rates.
  - Axes and values otherwise match the CSV.
- **fig1_one_dimensional_example.png.** Consistent with e4_shapes.log: (b) gives u''(1) = 1.05 at α=2 and −3.037 at α=4, against −3.037 prescribed. In (c) the plain slope at x=3 (−0.1137) is visually indistinguishable from the prescribed −0.1162, so only x=1 visibly shows the failure. No errors found.
- **fig_shape_functions_2d.png.** Plausible: the ∂y shape function is odd in y about the node. It is not referenced in RESULTS.md. No errors found.

## 6. Verdict

The tests pass, and most numbers in RESULTS.md match the data. The exactness classifications are robust to stencil radius and precision at the settings used. RESULTS.md is **not ready** to support a paper as written:

1. The headline exponent rule "iff r < α, with no exception" is **contradicted by the code's own setting**: mixed second derivatives are exact at α=2 in 2D and 3D. The aggregated CSV hid this.
2. "17 of 17 previously failing" is really 10 previously failing.
3. Three numbers are wrong: 5.1 → 4.4, 1.4 → 1.3, 3.6e-4 → 3.7e-4. The boundary-error statement is false for Franke.
4. The Hermite-vs-Lagrange accuracy claim is confounded. At equal m and α the gain is only about 1.5×.
5. The E5 figure label is wrong (100 vs 120 digits).

The D-invariance results (E2) and the completion result (E2b, extended in S3) hold up. The rule should be restated per datum, and its interaction with α should be tested.
