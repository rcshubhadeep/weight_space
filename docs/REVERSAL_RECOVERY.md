# Recovering a rank-one program from reversal pairs

**Question.** Can a compiler recover the two directions used by the previous hand-specified child from sentence labels, given the family of adjacent-word bilinear programs? We supplied 18 pairs, two sentences per pair. The compiler receives the texts and binary signs. It sees no word roles, seed adjective lists, marked swap locations, teacher outputs, or evaluation labels while writing the weights. The experimenter designed the intervention and the scoring family.

## Theorem and proof

Fix a word embedding e and define the matrix of a sentence x by summing e(left) e(right)^T over every adjacent word pair. Call this matrix F(x). A program W scores x by its Frobenius inner product with F(x). Work in real arithmetic.

Consider two nonadjacent internal slots, sharing the same following word. In each frame f, call the predecessor of the actual-outcome slot u_f and the predecessor of the expectation slot v_f. For each adjective pair (p,n), x+ puts p in the actual slot and n in the expectation slot. Its partner x- swaps those words. Assume identical remaining words in each pair. Then

    F(x+) - F(x-) = [e(u_f)-e(v_f)] [e(p)-e(n)]^T.

Proof. Every edge outside the four edges incident on the slots cancels. The two edges leaving the slots sum to (e(p)+e(n)) e(follower)^T in either ordering. The two incoming edges give the displayed outer product by distributivity. This argument requires the slots to have disjoint incident edges. With unequal followers h,k there is an additional outer product [e(p)-e(n)] [e(h)-e(k)]^T.

**Recovery.** Cross every frame with the same multiset of adjective pairs, once per combination. Sum the signed sentence matrices to form D. Distributing the two finite sums gives

    D = [sum_f (e(u_f)-e(v_f))] [sum_pairs (e(p)-e(n))]^T.

Thus rank(D) is at most one and is exactly one when both bracketed vectors are nonzero. In that case its singular directions are the normalized bracketed vectors up to simultaneous sign. For a unit Frobenius norm budget, Cauchy–Schwarz gives <W,D> <= ||D||_F, attained uniquely by D/||D||_F. This maximizes the **sum of paired score differences**. A threshold, individual correctness and transfer to unseen words require separate evaluation. Rank one alone can also occur with meaningless embeddings.

**Implementation.** We accumulate signed adjacency counts C in the small fit vocabulary. If the embedding columns satisfy E=QR, then D=ECE^T=Q(RCR^T)Q^T. An SVD of the small middle matrix yields the two directions, which we project over the parent's vocabulary. The deployed model sums adjacent-word products and thresholds at the mean fit score. Words are means of their subtoken projections. The float32 table and float64 threshold occupy 1,000,416 bytes plus the shared tokenizer. The inherited first two singular values were 18.00272 and 8.44e-16.

**Checks.** An exact symbolic-matrix test builds C from the actual 36 fitting sentences and compares every integer entry with the predicted outer product. It checks rank one and separately expands the unequal-follower correction. On all 32 reused service pairs the exported child's table satisfies the scalar swap identity using exact rational representations of the stored floats. The maximum deviation of the float32 score difference from the rational prediction was below 1e-8 on these inputs.

## Findings

| Evaluation | Inherited correct | Both reversals correct | Permuted vocabulary correct | Flipped training-pair orientations correct |
|---|---:|---:|---:|---:|
| Fit, 36 | 36 | 18/18 | 36 | 8 |
| Reused service, 64 | 64 | 32/32 | 32 | 26 |
| Food, 72 | 48 | 18/36 | 36 | 45 |
| Negated outcome, 64 | 12 | 0/32 | 34 | 36 |

The two controls each use one deterministic seed. The 72 food sentences cross two template families in both orders and three adjectives per polarity. They were specified before the first run of this compiler. The food result exceeds the flipped-label control by only three messages, with the same 18 complete pairs. All suites are synthetic. The service suite previously informed the choice of scoring family. Negation reverses the operational binary label when “anything but” precedes the actual outcome. All 32 negated pairs were ordered in the wrong direction. This makes scope a concrete failure to address.

## Reproduce

From the existing GPU research environment, using the model path defined in the [replication guide](ORDER_INHERITANCE.md):

```powershell
python -m examples.reversal_compiler --compile $model
python -m examples.reversal_compiler --message "I predicted horrible service but experienced wonderful service."
python -m unittest discover -s tests -v
python scripts/verify_evidence.py
```

[Frozen report with rankings, controls, and roundoff](../results/2026-11-25/reversal_compiler.json)
