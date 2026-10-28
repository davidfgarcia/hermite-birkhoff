# Referee report on "Generalized Hermite–Birkhoff interpolation ... via singular-weight moving least squares" (draft, Sections 3–5)

Scope: Sections 3, 4 and 5 of `paper/main.tex`. Sections 1, 2 and 6–8 are placeholders and were not reviewed. I re-checked every proof line by line. I also ran independent exact (sympy) and high-precision (mpmath, `hb.MPSingularMLS`) computations. No project file was modified. The scratch scripts are in `/tmp/claude-0/-home-claude/ecac4df8-638b-50d8-acc0-62781da0f3b8/scratchpad/`.

---

## 1. Summary verdict

**The core proofs hold up.** I found no error in any of these:

- Lemma 3.1 (Fischer adjointness)
- Lemma 3.2 (operator Taylor expansion)
- Lemma 3.3 (the floor criterion for Δ^ℓ ξ^β)
- Theorem 4.1 (structure near a node)
- Theorem 4.2 (error representation)
- Corollary 4.3 (sufficient condition)
- the sufficiency part of Corollary 4.4
- Proposition 4.6 (necessity at leading order)
- Proposition 5.1 (the KL counterexample)
- Proposition 5.2 (completion)

I verified Theorem 4.2 term by term in exact arithmetic. I independently re-derived the closed form of Proposition 5.1.

**The problems are in what the paper claims beyond its proofs.**

1. **Conjecture 4.7 is false as stated.** I give four kinds of counterexample, three of them exact and structural:
   - two nodes (K = 2) at any positions;
   - a configuration where only one other functional sees N_i;
   - a choice of the weight germs ρ_j;
   - a symmetric geometry.

   In each, condition (i) fails and yet the datum is reproduced for all data values. One of them is a Komargodski–Levin-type configuration (slopes at points without values) that **is** interpolated.
2. **Several sentences overstate the proven necessity:**
   - "(ii) is necessary";
   - "exactness iff α > k";
   - "exactly the multilinear derivatives";
   - "the smallest admissible exponent".

   Some of these become correct once Proposition 4.6 is invoked with its extra hypothesis. Others are not proven, or are misleading because reproduction is linear in q.
3. One factual misstatement: evenness of α is not "used exactly once".

Recommendation: **major revision.** The theorems can stay. The conjecture, the Hermite corollary and the necessity claims must be restated.

---

## 2. Errors (must fix)

### E1. Conjecture 4.7 is false as stated (most severe)

The conjecture says: "Under Assumption 3.4, and for data in general position, q ∈ Q_i is reproduced iff (i) and (ii)." The "only if (i)" direction fails in configurations that have nothing to do with measure-zero positions or data values.

**Why it can fail.** Theorem 4.2 shows that, when (ii) holds, the error is the middle term

  Σ_γ (1/γ!) D^γ[x ↦ [∂^γ q, ν(x)]_{x_i}](x_i).

This vanishes whenever the derivatives D^γ ν(x_i) are orthogonal to the relevant ∂^γ q. That happens, even when ∂^γ q ∉ Q_i, in at least these situations:

**(a) K = 2: one other node, any positions, any data.**
- Then B_x = c(x)·B with c(x) = w_j(x)/ρ_i(x) a scalar. So F depends on (x, t) only through the product t·c(x) (rescale s ↦ c s).
- Hence ν(x) = n(|x − x_i|^α c(x)), and D^γ ν(x_i) = 0 for all |γ| < α.
- For |γ| = α the middle term reduces to c(x_i)·[Δ^{α/2} q, n′(0)], because Σ_{|γ|=α} (1/γ!) D^γ(|x|^α)(0) ∂^γ = Δ^{α/2}.
- Consequence: **for K = 2, every datum of degree < α, and every datum q with Δ^{α/2} q = 0 and deg q ≤ α, is reproduced regardless of (i).**

Verified numerically (§5, tests N3 and N4). Each pair below is reproduced when K = 2 and fails when K ≥ 3:
- gradient-only node {D_x, D_y}, α = 2;
- D_xy in {f, D_x, D_xy}, α = 2;
- D_xx in {f, D_xx}, α = 4;
- the harmonic cubic ξ1²ξ2 − ξ2³/3 in {1, ξ1, ξ2, ξ1ξ2, ξ1²ξ2 − ξ2³/3}, α = 2;
- {D_x, D_y, D_xy}, α = 2;
- 1-D slope-only node, m = 1 and 2, α = 2 and 4 (exact rational arithmetic).

K = 2 is a legitimate instance of Assumption 3.4 (A3 holds). "General position" does not exclude it.

**(b) Only one other functional sees N_i (exact, all positions).**
- Setup: d = 1, m = 1. Data f′(0), f(1), f′(5/2). N_0 = constants, and only f(1) is non-zero on N_0.
- Then ∇_N B_x(p* + n_0) involves only the residual of f(1), which is zero at the optimum. So ∂_x ν(x_i) = 0.
- Exact result: u′(0) − y = 0, u′(5/2) − y = 0, u(1) − y = 0 for arbitrary rational data.
- This is a Komargodski–Levin-type configuration (derivatives at points carrying no values) **that is interpolated**.
- Adding a functional that sees N_i destroys exactness (control computations in §5, N5).

**(c) Weight germs.** Assumption (A1) allows any smooth ρ_j > 0. Take ρ_j = c_j|x − x_j|^α near x_i, modified smoothly and positively near x_j. Then w_j is constant near x_i and ν depends on x only through t(x). In the exact setting of Proposition 5.1 this gives u′(1) = 1, the prescribed value, instead of −1. The only change is ρ_j near x_i (§5, N6).

**(d) Symmetric geometry.**
- Setup: gradient-only datum D_x at the origin; all other nodes on the x2-axis. Then ∂_{x1} w_j(x_i) = 0.
- D_x is reproduced to about 1e-48–1e-96, for pure-power and Gaussian weights and for m = 1, 2 and α = 2, 4.
- Perturbing the nodes off the axis by 0.05 gives errors around 1e-2.
- Only this case is plausibly excluded by "general position".

**Fix.**
- Restrict the conjecture explicitly: ρ_j ≡ 1, or ρ_j in a generic class; K ≥ 3; and a non-degeneracy condition on the restricted problem. For example, "the functionals Λ_j|_{N_i}, j ≠ i, do not form an interpolatory system on N_i" and "the gradients ∇_x(w_j/ρ_i)(x_i) are in general position".
- Define "general position" precisely: generic in which parameters (node positions, ρ, data), and outside which set.
- Better: state a correct theorem now. "For K = 2, D^γ ν(x_i) = 0 for |γ| < α." This is easy to prove with the scaling argument above.
- Remove from the remark after the conjecture the claim that only a proof of necessity of (i) is missing. That necessity is false without such hypotheses.
- Note that the 175/175 agreement never included K = 2 or interpolatory restricted systems, which is exactly where the conjecture breaks.

### E2. "The next result shows that it cannot be improved: condition (ii) is necessary." Also Contribution (2): "the second condition is necessary"

Proposition 4.6 proves necessity only when:
- deg q = α;
- q satisfies (i);
- the other nodes are unisolvent by themselves (an additional hypothesis); and
- in the sense "there exist data at the other nodes".

Nothing is proven for deg q > α. The abstract's "(ii) cannot be dropped" is accurate. "(ii) is necessary" is not.

**Fix.** Write "condition (ii) cannot be dropped: it is necessary at leading order (Proposition 4.6)". In the contributions list, say "necessary for deg q = α".

### E3. Corollary 4.4(a), second clause, and Contribution (3)

The corollary says "the smallest admissible weight exponent is the smallest even integer exceeding k", and Contribution (3) says "exactness iff α > k for the full data set". The proof of Corollary 4.4 proves only the "if" direction. "Admissible" is never defined.

The "only if" direction does follow from Proposition 4.6 (proved later):
- If α ≤ k is even, then ξ1^α ∈ Q_i.
- That symbol satisfies (i) by D-invariance.
- It has degree α and Δ^{α/2} ξ1^α = α! ≠ 0.

But this needs the extra hypothesis of Proposition 4.6 (the other nodes unisolvent by themselves). It gives failure only "for some data at the other nodes" (I checked numerically that it fails even for K = 2; §5, N7).

**Fix.**
- Define "admissible" as: every datum at x_i is reproduced for all data values.
- Add the extra hypothesis to the statement.
- Move the sharpness part after Proposition 4.6, or forward-reference it, and give the two-line argument above.

### E4. Corollary 4.4(b): "for α = 2, exactly the multilinear derivatives are covered, whatever their order"

- **Sufficiency** is fine: Σ⌊β_j/2⌋ < 1 if and only if every β_j ≤ 1.
- **"Exactly" as an iff.** Non-reproduction of the non-multilinear D^β is proven only for |β| = 2, i.e. ξ_j², via Proposition 4.6 with its extra hypothesis. For |β| ≥ 3 (D_xxy, D_xxx, …) it is only observed numerically. My checks agree (§5, N1).
- **Misleading because of linearity.** The set of reproduced data is a linear subspace of Q_i, and it contains every q ∈ Q_i with Δq = 0. So D_xx − D_yy **is** reproduced at α = 2: the errors in D_xx and D_yy are identical to 10 digits, and their difference is exactly 0 (§5, N2). Saying that "exactly the multilinear derivatives" are reproduced suggests that no combination of D_xx and D_yy is reproduced.

**Fix.**
- Say "among the monomial data D^β, condition (ii) holds exactly for the multilinear ones".
- State the subspace description: the reproduced data contain {q ∈ Q_i : Δ^{α/2} q = 0}.
- Add the error identity that Theorem 4.2 gives for D-invariant Q_i: err(q) = [Δ^{α/2} q, τ]_{x_i}. This is a stronger and checkable statement. For example, at α = 2, err(D_xxx) = 3·err(D_xyy). It holds to 12 digits (§5, N2).

### E5. Remark 4.? (`rem:odd`): "Evenness of α is used exactly once"

This is false. Theorem 4.2 also uses evenness essentially: |z|^α = (|z|²)^{α/2} is a polynomial, which is what makes Lemma 3.1 applicable and gives the factor Δ^{α/2}. Without it the last term of (eq:error) has no meaning. The second sentence of the remark ("u is in general only finitely smooth") is asserted without proof.

**Fix.** Say "used twice: smoothness of t, and the Fischer identity in Theorem 4.2". Phrase the odd-α smoothness claim as an expectation or give an example.

### E6. Abstract: "a completion procedure that makes arbitrary Birkhoff data exact"

Proposition 5.2 gives exactness only for data with Δ^{α/2} q = 0, for example when α > deg q. The Proposition itself is stated correctly; the abstract and Contribution (4) are not.

**Fix.** Write "... makes any Birkhoff datum q with Δ^{α/2} q = 0 (in particular every datum, once α exceeds its order) exact".

---

## 3. Gaps (should fix)

### G1. Theorem 4.1 (checked; correct)

I checked each step:
- The splitting J_x/w_i = |Lr|² + t B_x, where w_j/w_i = w_j|x − x_i|^α/ρ_i.
- The gradient equations, and the substitution r = t s, which is legitimate because t > 0 off x_i.
- The Jacobian. At t = 0, ∂F₁/∂s = 2L*L, ∂F₂/∂s = t(∇²B)_{NR} = 0, and ∂F/∂n contains the Hessian blocks. It is block upper-triangular with invertible diagonal blocks.
- Strict convexity on p* + N_i, via (A3) together with w_j(x_i) > 0. The latter holds because x_i ≠ x_j and ρ_j > 0. Say this explicitly.
- The identification of the IFT solution with p_x. It uses only uniqueness of the critical point of the strictly convex J_x, not the uniqueness part of the IFT. This is valid.
- The smoothness claims.

Minor points:
1. State that U is open and that ρ_i(x) > 0 on U is what makes B_x smooth.
2. "u extends to C^∞(Ω)" requires applying the theorem at every node. Say "applying this at each node".
3. p_i^*, ν and σ depend on the chosen inner product. Only ν(x_i) and the total error are canonical. Worth a sentence, since Proposition 4.6 relies on this choice.
4. The heading-level statement "a polynomial p_i^* with λ_{i,s}(p_i^*) = y_{i,s}" should add "p_i^* ∈ R = N_i^⊥". Proposition 4.6 uses this.

### G2. Theorem 4.2 (checked; correct)

- The diagonal chain rule is correct.
- Lemma 3.2 applies, since D_x and D_y commute.
- Setting y = x_i commutes with D_x trivially, because y is an independent variable.
- The truncation argument is correct: |z|^α(τ − Tτ) = o(|z|^{deg q}), and deg Δ^{α/2} q ≤ deg q − α ≤ deg Tτ.

Small points:
- Say explicitly that u(z) = Φ(z, z) holds also at z = x_i, by the definition of the extension.
- Say that the γ = 0 term of the middle sum is [q, ν(x_i)]_{x_i}.

**Exact check.** In d = 1, at a slope-only node with three value nodes, I evaluated all three terms of (eq:error) in rational arithmetic for (m, α) = (2, 2), (3, 2), (3, 4) and q = ξ^0, …, ξ^m. Left- and right-hand sides agree exactly in all 11 cases (§5, N8).

### G3. Proposition 4.6 (checked; correct, can be strengthened)

The argument is correct, and the inner product is used consistently:
- R = N_i^⊥ in ⟨·,·⟩.
- K is built with adjoints in the same inner product.
- K(n_0 − v) ⊥ N_i, which is equivalent to K(n_0 − v) ∈ R.
- v ↦ w = v − n_0(v) is the K-orthogonal projection onto W = K^{-1}R.
- Hence K(W) = R, and s_0 = (L*L)^{-1}K w ranges over R.
- 1 ∈ Q_i: some ∂^γ q with |γ| = α is a non-zero constant.

Suggestions:
1. The set of bad v is a proper linear subspace, so failure holds for **all data outside a hyperplane**, i.e. generically. State it that way.
2. The hypothesis "other nodes unisolvent by themselves" is only used to make K positive definite. Positivity on the K-relevant subspace probably suffices. Either weaken it or remark on why it is needed.

### G4. Proposition 5.2

- The hypothesis deg Q̄_i ≤ m is redundant: differentiation lowers degree, so deg Q̄_i = deg Q_i ≤ m.
- Note that y_q for q ∈ Q_i is unchanged by the augmentation, because the representation in the augmented basis is unique. The proof uses this implicitly.
- (A2) and (A3) are indeed preserved.

### G5. Section 5.1 framing

- The sentence "Condition (i) fails for such data ... the following exact computation shows that the claim is indeed false" reads as if failure of (i) implied failure of reproduction. E1(b) shows a KL-type configuration that **is** reproduced. One explicit counterexample suffices to refute a universal claim, so Proposition 5.1 stands. Reword to "can fail, and does in the following example".
- Display the normal equations, as the TODO already says.
- Check against the KL paper:
  - that their setting allows ρ ≡ 1, m = 1, α = 2 with separate value and derivative points;
  - that "the smallest space allowed" is accurate;
  - that their conjecture is "α = 2k suffices".

  I could not verify these without the paper. Note also that for k = 1 the "sharpening" α = k + 1 = 2 = 2k is no sharpening at all.

### G6. Remark after Definition 3.5 (individual weights)

- "Without change" is roughly right for per-functional positive factors ρ_{i,s}. The first block then becomes 2L*W(x)L, which is still invertible.
- For "a weight that is singular only for some of them", Q_i and N_i must be redefined to contain only the singular functionals. The other functionals at x_i move into B_x.

Say this, or drop the remark.

---

## 4. Wording and overstatements

1. Contribution (2): "the second condition is necessary". Replace with "necessary at leading order" (E2).
2. Contribution (3) and Corollary 4.4(a): "exactness iff α > k", "smallest admissible". Add the hypotheses and a proof (E3).
3. Corollary 4.4(b): "exactly ... are covered". Restate in terms of the subspace (E4).
4. The remark "used exactly once" (E5).
5. Abstract: "makes arbitrary Birkhoff data exact" (E6).
6. "For data in general position" is undefined (E1).
7. "Corollary 4.4(a) proves, in a sharper form, the conjecture of KL that α = 2k suffices". It is sharper only for k ≥ 2, and should be checked against the source (G5).
8. "already α = k+1 (or k+2, to make it even)". Say instead: "α = k+1 for odd k, α = k+2 for even k".
9. The remark after the conjecture: "The missing parts of a proof are ... necessity of (i)". That necessity is false without extra hypotheses (E1).

---

## 5. Numerical checks performed

All scripts are in `/tmp/claude-0/-home-claude/ecac4df8-638b-50d8-acc0-62781da0f3b8/scratchpad/`:
- `swimls1d.py` is an independent exact 1-D SWIMLS. It does not use `hbmls`: it forms J_x with weights multiplied through by Π_j (x − x_j)^α and solves with sympy.
- The 2-D tests use `hb.SingularMLS` and `hb.MPSingularMLS(op).derivative(point, beta, y, r=1e-6)`, whose automatic precision is ≥ 100 digits.

Every run took under a minute.

### N0. Proposition 5.1, independent re-derivation (`kl.py`)

```
u = x*(x - 2)*(x - 1)*(x**2 - 4*x + 5)/(2*x**4 - 14*x**3 + 37*x**2 - 42*x + 19)
matches paper: True
u(0)= 0   u(2)= 0   u'(1)= -1   u'(3)= 31/25
real roots of denominator: []  min on R ~ 1.83
m=1 alpha=4: u'(1)=-3, u'(3)=1789/1681
m=2 alpha=2 (extra value f(4)=0): u'(1)=u'(3)=-347/289
m=2 alpha=4 (extra value f(4)=0): u'(1)=u'(3)=-30911/9409
```

The proposition is confirmed exactly. The failure persists for α = 4 and for m = 2.

### N1. Condition (i) holds, (ii) fails, deg q > α: 2-D Hermite k = 3, α = 2, datum D_xxy (`test2d_b.py`)

Errors for D_xxy, D_xy and D_xxx, with two random data vectors per row:

```
Hermite k=3, alpha=2, m=3, K=3 (others k=1)   +9.14e+00  +0.00e+00  +3.29e+01 | +3.50e+01  +0.00e+00  +1.50e+01
Hermite k=3, alpha=2, m=3, K=2 (other k=3)    -5.33e-01  +0.00e+00  +1.48e+01 | -4.53e+00  +0.00e+00  -3.71e+01
Hermite k=3, alpha=4, m=3, K=3 [control]      all 0
```

This agrees with the conjecture: D_xxy fails even with K = 2, which matches the |γ| ≥ α analysis in E1(a).

### N2. Linear structure predicted by Theorem 4.2 (`ratio.py`, `dxxmyy.py`)

For a D-invariant Q_i, err(q) = [Δ^{α/2} q, τ]. At α = 2 this predicts:
- err(D_xx) = err(D_yy);
- err(D_xxx) / err(D_xyy) = Δξ1³ / Δξ1ξ2² = 6ξ1 / 2ξ1 = 3.

```
alpha=2 m=3: err Dxx=+4.7563469403e+00 Dyy=+4.7563469403e+00 Dxy=+5.1e-47
   Dxxx/Dxyy = 3.000000000000   Dyyy/Dxxy = 3.000000000000
alpha=2 m=4: err Dxx=+7.9047976563e+00 Dyy=+7.9047976563e+00
   Dxxx/Dxyy = 3.000000000000   Dyyy/Dxxy = 3.000000000000
Hermite k=2, alpha=2, K=5: err Dxx=-3.426282e+01, err Dyy=-3.426282e+01, err(Dxx-Dyy)=+0.00e+00
```

This is strong confirmation of Theorem 4.2. It also shows that D_xx − D_yy is reproduced at α = 2 (E4).

### N3. Condition (ii) holds, (i) fails: K = 2 versus K ≥ 3, 2-D (`test2d.py`, `test2d_b.py`, `xy.py`)

```
grad-only {Dx,Dy}, K=2, m=1,a=2                  0.00e+00  0.00e+00      <- reproduced
grad-only {Dx,Dy}, K=3, m=1,a=2                 -1.06e+00  1.94e-01
{f,Dx,Dxy} datum Dxy, K=2, m=2,a=2               0.00e+00                <- reproduced
{f,Dx,Dxy} datum Dxy, K=4, m=2,a=2              -1.02e-01
{f,Dxx} datum Dxx, K=2, m=2,a=4                  0.00e+00                <- reproduced
{f,Dxx} datum Dxx, K=2, Gaussian delta=1, a=4    0.00e+00                <- reproduced
{f,Dxx} datum Dxx, K=4, m=2,a=4                 -3.15e+00
{1,x,y,xy,x^2y-y^3/3} datum h, a=2, m=3, K=4    -1.92e+00 | +5.82e+00
{1,x,y,xy,x^2y-y^3/3} datum h, a=2, m=3, K=2    -1.07e-15 | +1.11e-16    <- reproduced (float-level)
  same, K=2, Gaussian delta=0.8                 +4.44e-16 | +3.33e-16
{Dx,Dy,Dxy} (Q has no 1), a=2, m=2, K=2         ~1e-48 (all three)        <- reproduced
{Dx,Dy,Dxy}, K=4                                 7.4e-01 -1.45e+00 1.74e+00
```

These are counterexamples to Conjecture 4.7 (E1a).

### N4. Exact 1-D, K = 2 (`k2.py`)

Node 0 carries only the slope.

```
m=1 a=2 node1={f,f'}            : u'(0) - y = 0
m=2 a=2 node1={f,f',f''}        : u'(0) - y = 0
m=2 a=4 node1={f,f',f''}        : u'(0) - y = 0
m=2 a=2 node0={f',f''}          : u'(0) - y = 0,  u''(0) - y = 136/9   (deg q = alpha: fails, as predicted)
control K=3 m=1 a=2             : u'(0) - y = -9296/2523
```

### N5. Exact 1-D, only one other functional sees N_i (`k3sq.py`, `extra.py`)

```
K=3 m=1 a=2 {f'(0), f(1), f'(5/2)}: u'(0)-y = 0, u'(5/2)-y = 0, u(1)-y = 0     <- KL-type data, interpolated
K=4 m=2 a=2 {f'(0), f(1), f(-2), f'(3)}: u'(0)-y = -21296/12615
control (extra value at 3):              u'(0)-y = -196750/77841
```

### N6. Weight germs (`rho.py`)

This is the Proposition 5.1 configuration with w_j constant near x = 1 for j ≠ i, i.e. ρ_j = c_j|x − x_j|² near 1:

```
u'(1) with w_j constant near x=1: 1   (prescribed 1)
```

### N7. Proposition 4.6 with K = 2 (`prop46.py`)

Hermite k = 2, α = 2, one other Hermite node:

```
err Dxx=-8.8540e-01  Dyy=-8.8540e-01  Dxy=+2.4e-47
err Dxx=-1.4190e+00  Dyy=-1.4190e+00  Dxy=-1.3e-47
```

This agrees with Proposition 4.6.

### N8. Exact term-by-term check of (eq:error) (`errrep.py`)

Setup: x_i = 0 with Q_0 = span{ξ}; value nodes at 1, −7/4 and 5/2; Euclidean inner product on coefficients. Then N = span{1, Y², …}, R = span{Y}, p* = y0·Y, ν = the non-Y coefficients, σ = ((a_1 − y0)/x^α)·Y and τ(z) = σ(z)(z).

```
m=2 alpha=2 q=xi^0..2 : lhs = rhs (3/3)
m=3 alpha=2 q=xi^0..3 : lhs = rhs (4/4)
m=3 alpha=4 q=xi^0..3 : lhs = rhs (4/4)
```

### N9. Symmetric geometry (`testA.py`)

D_x at the origin, other nodes on the x2-axis. Errors for two random data vectors:

```
m=1 a=2 pure power, values on axis       : -1.0e-95, 0.0
m=1 a=2 Gaussian                         : ~3e-96
m=2 a=2 / a=4, Hermite k=2 on axis       : ~1e-48
same, nodes perturbed by 0.05            : 1.2e-02, 1.7e-02 / 1.4e-02, -2.6e-03
```

### On "a symbol whose derivative leaves Q_i invisibly on P_m"

This cannot happen. Because deg ∂^γ q ≤ m and the pairing is non-degenerate on P_m, ∂^γ q ∉ Q_i if and only if there is some n ∈ N_i with [∂^γ q, n]_{x_i} ≠ 0.

The "invisibility" that does occur lives in the **range of the derivatives of ν** (E1). An example is {D_x, D_y, D_xy} with K = 2. There ∂^{(1,1)}(ξ1ξ2) = 1 ∉ Q_i, but D^{(1,1)}ν(x_i) ∝ D^{(1,1)}|x|² = 0.

---

## 6. Suggestions

1. Replace Conjecture 4.7 with a proven statement and a corrected conjecture:
   - **(a) Theorem (K = 2).** D^γ ν(x_i) = 0 for |γ| < α, and the |γ| = α part of the middle term equals c·[Δ^{α/2} q, n′(0)]. So for K = 2 a datum with deg q ≤ α is reproduced iff Δ^{α/2} q = 0, or its top part satisfies (i).
   - **(b) Conjecture.** For ρ ≡ 1 (pure power weights), K ≥ 3, node positions outside a proper algebraic subset, and restricted systems {Λ_j|_{N_i}} that are not interpolatory: reproduced iff (i) and (ii).
2. Promote the linear error functional q ↦ [Δ^{α/2} q, τ]_{x_i} (valid for D-invariant Q_i) to a corollary. It gives exact, testable predictions such as err(D_xx) = err(D_yy) and err(D_xxx) = 3 err(D_xyy) at α = 2. It also gives the subspace of reproduced data. This is more informative than the monomial criterion in Corollary 4.4.
3. Prove necessity of (i) at first order under a non-degeneracy hypothesis. When |γ| = 1 < α, the middle term is
   [∂_j q, ∂_{x_j} n(x_i, 0)],
   where ∂_x n(x_i, 0) = −H_NN^{-1} Σ_j ∇(w_j/ρ_i)(x_i) Λ_j^*(residual_j).
   This vanishes in exactly the degenerate situations of E1. It is the natural hypothesis to impose.
4. Strengthen Proposition 4.6 to "for all data at the other nodes outside a hyperplane".
5. In Section 6, add K = 2 cases, symmetric configurations and interpolatory restricted systems to the test suite. The current 175/175 agreement samples only the generic regime.
6. Drop the redundant degree hypothesis in Proposition 5.2. In Corollary 4.4, write the necessity direction explicitly via Proposition 4.6.
