# Protected rank-one patches

The proofs below concern a linear layer and the downstream computation at a fixed answer position. Held-out behavior is measured separately. The implementation uses finite precision.

## A removable branch

For a down-projection $W$, input $x$, output direction $u$, key $v$, and scale $s$, the distributive and associative laws give

$$
(W+suv^T)x=Wx+s u(v^T x).
$$

This allows a removable branch on a frozen checkpoint. A patch has two factors and can still change many entries when merged. Materializing a merged FP16 matrix can round differently from computing the branch in float32 before casting.

## Protection on an activation span

Let $S$ span the protected activations, and let $P=I-QQ^T$ with $Q$ an orthonormal basis covering $S$. For target activations stored in rows of $X$ and desired shifts $b$, set $Z=XP$. [The solver](../weight_space/low_rank.py) computes

$$
v=Z^T(ZZ^T+\lambda I)^{-1}b.
$$

It solves the ridge normal equations because $(Z^TZ+\lambda I)Z^T=Z^T(ZZ^T+\lambda I)$. Positive $\lambda$ makes the solution unique. Since $Z^T=PX^T$, the key lies in the range of $P$. The branch therefore vanishes for each input in $S$ in exact arithmetic.

LFM3's output direction $u$ is the normalized difference of the two answer-token unembedding rows. The desired shifts come from finite differences through the final normalization and readout, with regularization and clipping. The finite-difference fit is an approximation. The orthogonality statement is independent of its accuracy.

## Composing independently solved patches

At the same down-projection, the branches add in exact arithmetic. Suppose patch A's key is fitted with task B's anchor activations in its protected span and vice versa. On an A anchor, B's contribution is zero, so the combined projection equals the A-only projection. The argument is symmetric on B anchors. Because there is no later token mixing at the measured position in this experiment, the same equality applies to that position's downstream output in exact arithmetic. This claim applies to the fitted spans and fixed inputs. Autoregressive generation may change later inputs and earlier-layer or different-layer edits may change the activations themselves.

The experiment uses the same output direction for both tasks because they share A/B answer tokens. Their combined branch therefore still has rank at most one, with effective key equal to the sum of the scaled keys. The anchor argument also holds for different output directions at the same projection, which this run does not evaluate.

For any new input $x$, Cauchy–Schwarz gives $|v^T x| \leq \|v\|_2\operatorname{dist}(x,S)$. This explains why the protection can weaken as held-out inputs depart from the protected span. Branch scales multiply the bound.

## A testable bound for an unseen answer

The experiment calculates the actual norm $r$ of the second patch's contribution at the measured activation and bounds its effect on the first patch's answer margin. In dimension $d$, RMS normalization without the learned gain maps $z$ to $\sqrt d\,z/\sqrt{\|z\|^2+d\epsilon}$. Its Jacobian has operator norm at most $\sqrt d/\sqrt{\|z\|^2+d\epsilon}$. Along any segment within radius $r$ of the center $z_0$, use $\max(\|z_0\|-r,0)$ as a lower bound on the norm. Multiplying by the maximum absolute learned gain bounds the next perturbation radius. At the final normalization, the gain can be folded into the answer-row difference before applying Cauchy–Schwarz.

To see the Jacobian bound, write $q=\sqrt{\|z\|^2+d\epsilon}$. The Jacobian is $\sqrt d\,(I/q-zz^T/q^3)$. Its eigenvalue in directions orthogonal to $z$ is $\sqrt d/q$, and its eigenvalue parallel to $z$ is $\sqrt d\,d\epsilon/q^3$, which is no larger. At $z=0$ the same bound follows directly. Integrating the Jacobian along the segment gives the change bound, and the reverse triangle inequality supplies the segment-wide lower bound on $\|z\|$.

If the magnitude of the first patch's answer margin exceeds this bound, the second patch cannot flip that forced choice in real arithmetic. An inconclusive bound allows either outcome. The code evaluates the formula numerically and checks the FP16 predictions separately. It provides no certified floating-point rounding bound.

[Algebra tests](../tests/test_low_rank.py) check dense/branch equivalence and the protected solve. The [composition experiment](../examples/lfm_patch_composition.py) measures raw cross-patch contributions, actual flips and the fraction for which the sufficient margin inequality holds.

## Exact decision intervals for one patch strength

With the first patch fixed, let $z(t)=z_0+t z_1$ be the raw final feed-forward output,
$r$ the residual, $g$ the post-normalization gain, and $a$ the difference of the
two answer rows multiplied elementwise by the final normalization's gain. Write
$q(t)=\|z(t)\|^2/d+\epsilon$. After removing the positive final normalization
denominator and multiplying by $\sqrt{q(t)}$, the margin has the same sign as

$$
f(t)=(a^T r)\sqrt{q(t)} + a^T(g\odot z_0) + t\,a^T(g\odot z_1).
$$

Thus $f(t)=R\sqrt{C+Dt+Et^2}+U+Vt$. The radicand is strictly positive because
$\epsilon>0$. To check preservation throughout an interval starting at zero,
orient $f$ so that $f(0)>0$. If $R=0$, the minimum of the linear term is at an
endpoint. With $R>0$, only the part of the interval where $U+Vt<0$ can fail.
There we need the minimum of $R^2q(t)-(U+Vt)^2$ to be strictly positive.
With $R<0$, $U+Vt$ must stay positive and the negated quadratic must have a
strictly positive minimum. Evaluate the quadratic at each endpoint and at its
vertex if the vertex is an interior minimum. Checking the signs before
squaring removes extraneous roots. An initial tie or a tangency inside the
interval fails strict preservation, even if both endpoint answers agree.

The [implementation](../weight_space/decision_boundary.py) uses exact rational
coefficients computed from the binary values of captured tensors and keys.
Bisection with this interval predicate brackets the first tie from both sides.
This is an exact-arithmetic check for the downstream model with those values
held fixed. Upstream computations and captured values were produced in finite
precision. Live FP16 normalization and dot products can disagree with the
exact-arithmetic decision.

The optional clipping diagnostic chooses the full strength for prompts whose
whole interval passes. For the rest, it uses 63/64 of the lower bracket of
the first tie. It measures the FP16 downstream choice against the first patch
alone. The checked backoff variant tries that strength, halves failing
strengths up to eight times, and falls back to zero. A zero-change count for
this variant is expected because it explicitly checks the cached rounded
decision. Mean retained strength measures what the check costs. These are
retrospective diagnostics on the cached suffix, rather than a deployed
runtime policy or a proof about all possible FP16 values in the interval.
[Symbolic tests](../tests/test_decision_boundary.py) use SymPy to verify the
reduction and a first-tie bracket, including an interior flip that endpoint
checks miss.
