# Second independent review (whole draft), Oct 28 2026

Referee-style read of the complete draft (Sections 1–8 and appendices), after the hand-off of
paper 1. Recommendation: **minor revision**; no false theorem. Earlier `REVIEW.md` covered only
Sections 3–5 of an older version.

## Must-fix items and how they were resolved

| # | Finding | Resolution |
|---|---|---|
| 1 | Example on harmonic data (Toan et al. schemes) was vacuous: with #data = dim Π_m and unisolvent data, SWIMLS returns the polynomial interpolant and every datum is reproduced trivially. | Example 4.8 now says so, and states the informative case (more data than dim Π_m). |
| 2 | Theorem "frozen minimiser" was much weaker than its proof: with Z constant, every q ∈ Q_i satisfies error = [Δ^{α/2}q, η], whatever its degree. The experiments text wrongly blamed deg q > α for failures. | New Theorem 4.11 (error representation with Z in place of ν, η = (g+σ)(z), ν = Z + t g). Theorem 4.12 now covers every q ∈ Q_i; Proposition 4.15 (first-order error) extended to Δ^{α/2}q = 0. Experiments text corrected. Reviewer's numerical check: harmonic cubic at α = 2 with K = 2 reproduced to 1e-32; K = 3 control fails. |
| 3 | Conjecture needed a precise form. | New Proposition 4.16 (dichotomy for pure power weights: always reproduced, or not reproduced for node positions outside a proper algebraic set). Conjecture 4.17 restated through it; remark explains how the four counterexample families of `REVIEW.md` relate (all excluded by D-invariance; explained by Thm 4.12/Cor 4.13, Prop 4.15). |
| 4 | "Completion repairs all 13 non-reproduced data" miscounted. | Now: the 13 data that fail because of (i); the two that fail because of (ii) cannot be repaired. |
| 5 | Shepard comparison overclaimed: the Δ^{α/2} refinement also holds for Shepard–Taylor with even exponent. | Reworded; what fitting adds is the invisible part ν and condition (i). Gordon–Wixom (1978) added. |
| 6 | Remark 4.2 (Backus–Gilbert) needs (A3); smoothness should be credited to Levin's argument. | Done; Contribution (1) reworded. |
| 7 | Abstract omitted the norming condition; Contributions (3)–(5) imprecise. | Done. |
| 8 | Bibliography entries to verify. | All verified (see README); added Armentano 2001, Wendland 2001, de Boor–Ron 1992, Stein–Weiss 1971. Levin 1998 weights checked: C^∞ compactly supported (his (2.15)). |
| 9 | Numerical text: table bound, list of orders, weights used, completion fit description. | Fixed (≤ 1.1e-28; k ∈ {2,3,4,5,7}; pure power weights stated in §7.2–7.3; Gaussian-weighted MLS fit in §5 and §7.3). |

## Nice-to-have items done

"Along the diagonal" in Remark 3.2; ∂_{x_1}(w_j/ρ_i) in the symmetric case; Appendix B
"up to degree β_j + 5"; m = 5 rates split for u and gradient; "does not converge at the predicted
rate (1.2 and 0.4)"; Halton separation ratio reported (0.56 → 0.14, so not uniformly separated;
the periodic grids are the clean test); boundary-value paragraph cut to one sentence; repository
link; Remark 6.2(d) constant renamed; unused \todo macro removed.

## Not done (optional)

Systematic notation clean-up (n, K, R/L, s, r, ℓ, h used in several roles); a table for the rates
in §7.4; a comparison with a competing Hermite–Birkhoff method (Hermite RBF or Shepard–Hermite–
Birkhoff).
