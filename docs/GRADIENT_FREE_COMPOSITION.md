# Two tasks, one gradient-free weight edit

**Supervised rank-one patch composition in LiquidAI LFM3-1.7B**  
Technical report · 25 November 2026

## Abstract

Can two useful edits to a language model coexist in a single always-active patch? We fitted separate rank-one edits to the final feed-forward layer of LFM3-1.7B for passage question answering (BoolQ) and movie-review sentiment (SST-2). Both tasks used A/B answer tokens. Fitting required labeled examples, forward passes, and a linear solve. The procedure used no backpropagation, gradient-based weight updates, or external teacher. Adding the two individually calibrated edits at full strength reduced SST-2 accuracy below the unedited model. Choosing their relative strengths on training-split calibration data produced one 39,056-byte patch that raised BoolQ prompt accuracy from 57.0% to 65.0% and SST-2 accuracy from 87.4% to 89.4%. The patch retained 55% and 86% of the respective single-task gains. These results establish that partial joint retention is possible on one checkpoint under a shared answer format. Direct-label controls show that this format can substantially affect the apparent benefit of editing.

## 1. What was learned, and how?

A reader could reasonably ask how a model improves without a gradient update. For any input to the edited layer, the patch asks one learned question: how far does this activation point along a fitted direction? The answer is a scalar. The patch uses that scalar to push the layer output along a second, fixed direction derived from the model's A and B answer-token weights. Different inputs therefore receive different corrections.

Write the original down-projection as $W$, its input as $x$, the fixed output direction as $u$, and the learned input key as $k$. The deployed computation is

$$
Wx + u(k^T x).
$$

We read activations at the final response position in block 23. For each labeled fitting example we ask how much its current answer margin falls short of a target margin of two logits. Two forward perturbations along $u$ estimate how sensitive the downstream margin is to a shift in that direction. A regularized, clipped inversion converts the shortfall into a target scalar shift. A ridge least-squares solve fits one key across the examples. For BoolQ the solver projects away a span of sentiment and urgency anchor activations. The SST-2 key uses the unprojected solve.

**What “gradient-free” means here.** No backward pass through LFM, parameter-gradient computation, Adam step, or fine-tuning epoch is used to produce either edit. Finite differences estimate a local directional sensitivity through the final normalization and readout. The keys are supervised fitted parameters. Calibration also uses labels to select scales. Calling this “no training” would obscure both sources of supervision. The model runs with the extra branch attached, and the original down-projection's digest was unchanged after evaluation. [Solver and derivation](METHOD.md) · [Implementation](../examples/lfm_weight_edit.py)

## 2. Why two edits fit in one patch

Let $b$ and $s$ be the BoolQ and SST-2 keys after their individual scale selections. Both use the same $u$ because the answer tokens and edited layer are identical. For any real coefficients $\alpha,\beta$,

$$
\alpha u b^T+\beta u s^T=u(\alpha b+\beta s)^T.
$$

**Proof.** Transposition and the outer product with a fixed left factor distribute over addition. The image of the resulting matrix lies in the span of $u$, so its rank is at most one. Thus a single effective key suffices at inference. This identity holds in real arithmetic. The deployed implementation forms the combined key and rounds the branch contribution along the FP16 forward path.

The identity guarantees a storage bound. It provides no lower bound on accuracy: on a particular review, either key can push the answer across its decision boundary. The full-strength experiment below measures that failure directly.

## 3. Experimental design

| Choice | BoolQ | SST-2 |
|---|---|---|
| Label signal for fitting | 128 training questions scored under both answer mappings | 512 labeled training strings |
| Scale and composition calibration | 128 additional questions, 256 prompts | 256 additional training strings |
| Evaluation unit | 3,270 public development questions, each with reversed and original A/B mappings | 872 public validation reviews, fixed negative=A and positive=B |
| Additional check | Count questions correct under **both** mappings | Compare against a scalar threshold on the unedited margin |

BoolQ sampling excluded training passages with exact development-passage matches and used distinct passages for fitting and calibration. SST-2 sampling excluded exact normalized validation strings and duplicates across fitting and calibration. Similar phrases may survive exact matching. Both public validation splits had been examined in earlier experiments. This composition run selected its coefficients using training-split calibration data.

We evaluated all 25 pairs from $\{0,0.25,0.5,0.75,1\}^2$ on calibration. For each task, we divided the candidate's gain over base by that task's single-edit gain over base. We maximized the smaller of these two fractions. Ties favored the larger sum of fractions, then the smaller sum of coefficients. Each task therefore had a say in the choice even though their baseline accuracies differed.

Nine of the 25 candidates improved both calibration scores. The full-strength sum and the selected combination each got 172/256 BoolQ calibration prompts right. They differed by **57 answers** on SST-2 calibration: 172/256 for the full sum versus 229/256 for the selected combination. The selection rule chose $0.25b+1.0s$.

## 4. Results

Every row below uses the same checkpoint, prompts, two-row readout, and validation inputs within a task. BoolQ's two prompts per question are paired observations.

| Always-active edit | BoolQ correct prompts / 6,540 | BoolQ correct in both mappings / 3,270 | SST-2 correct reviews / 872 |
|---|---:|---:|---:|
| Unedited | 3,729 (57.0%) | 601 (18.4%) | 762 (87.4%) |
| BoolQ alone | 4,682 (71.6%) | 1,748 (53.5%) | 719 (82.5%) |
| SST-2 alone | 4,109 (62.8%) | 923 (28.2%) | 783 (89.8%) |
| Full-strength sum | 4,136 (63.2%) | 966 (29.5%) | 726 (83.3%) |
| **Calibration-selected combination** | **4,254 (65.0%)** | **1,068 (32.7%)** | **780 (89.4%)** |

**Interference was asymmetric.** The sentiment patch alone added 380 correct BoolQ prompt decisions. The BoolQ patch alone cost 43 correct sentiment decisions. Adding both at full strength left sentiment 36 reviews below base. These cross-task controls matter: some of the selected patch's BoolQ gain was already available from the sentiment patch alone.

**The selected combination improved the trade-off.** Relative to the unedited system it added 525 net correct BoolQ prompt decisions and 18 net correct reviews. Relative to the sentiment patch alone, it added 145 net correct BoolQ decisions at a cost of three correct reviews. Relative to the full-strength sum it gained on both tasks. Inference used one shared key and output direction for every prompt, with no task router.

Validation replayed the frozen model's downstream computation from cached activations. An independent live-forward check agreed on all 51 audited choices (19 BoolQ prompts and 32 reviews). The audit samples came from the first and last chunks. The experiment did not run all combined decisions through the ordinary full-vocabulary head. [Frozen calibration grid, scores, and audit counts](../results/2026-11-25/boolq_sst2_composition.json)

## 5. What explains the gain?

Two earlier controls narrow the interpretation.

- **BoolQ answer mapping.** The individual BoolQ patch reached 71.6% under counterbalanced A/B. The best of 32 equal-norm random branches reached 64.1%. Applying the A/B-fitted branch to direct Yes/No answers changed 80.7% to 80.9% on a 512-question probe, one additional correct answer. The evidence favors improved use of the answer mapping. A general increase in reading comprehension remains unestablished. [BoolQ controls](../results/2026-11-25/boolq.json)
- **SST-2 decision threshold.** In a separate full-vocabulary replay, the individual A/B patch improved 760/872 to 785/872, correcting 53 answers and breaking 28. A calibrated threshold alone scored 759. With direct sentiment tokens and a separately fitted patch, the base scored 735, the patch 763, and the threshold alone 783. This control makes a practical difference to which intervention one would choose. FP16 readout and batch configuration differ from the composition table. [Full-head replay](../results/2026-11-25/sst2_liquid_edit.json)

The combined patch was evaluated with A/B on both tasks. The individual controls motivate testing its behavior under new readouts. They do not measure such transfer for the combination. This study covers one checkpoint, one pair of tasks, and one fitting split per task. Aggregate counts describe these validation sets. The frozen composition report lacks item-level paired outcomes for uncertainty estimates of the joint gains. Free generation, unrelated capabilities, and a third independent task remain unmeasured.

## 6. Relation to previous work

ROME demonstrated rank-one feed-forward edits for factual associations. Task arithmetic studied adding and subtracting parameter differences derived from task-specific models. TIES-Merging explicitly investigated interference among merged updates. The weight-space-lp project provided a direct motivation for computing behavioral edits with a numerical solver. Our experiment studies the particular case of supervised keys fitted directly against a common output direction, with measured cross-task interference and a calibration-only compromise. The rank identity is elementary linear algebra. We make no priority claim for rank-one editing or weight-space addition. [ROME](https://arxiv.org/abs/2202.05262) · [Task arithmetic](https://arxiv.org/abs/2212.04089) · [TIES-Merging](https://arxiv.org/abs/2306.01708) · [weight-space-lp](https://github.com/nathanrs/weight-space-lp)

## 7. Replication and the next test

We used a single RTX 2080 Super Max-Q with 8 GiB VRAM, CUDA FP16 inference, and the pinned LiquidAI/LFM3-1.7B checkpoint at revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`. The manifest records dataset revisions and input hashes. Starting from the repository root, reproduce the individual BoolQ fit and the fixed-A/B SST-2 edit, then run the composition script. Its report contains all coefficient trials, validation counts, and the original matrix digest. Verify the frozen evidence with `python scripts/verify_evidence.py`. [Environment and BoolQ commands](REPLICATION.md) · [SST-2 commands](SST2.md) · [Composition command](BOOLQ_SST2_COMPOSITION.md) · [Manifest](../results/2026-11-25/manifest.json)

The next experiment should separate *new-task learning* from *reuse of the A/B interface*. Freeze this patch first. Measure its zero-shot effect on a third binary task in both A/B and native-label forms. Then fit a third edit using training data, choose coefficients without validation access, and score all three tasks together. Record per-item predictions and evaluate retention on the unit of independent sampling. A result that survives a change of answer vocabulary would support a broader claim than the present two-task result.

**Benchmark sources.** [BoolQ, Clark et al.](https://aclanthology.org/N19-1300/) · [Stanford Sentiment Treebank, Socher et al.](https://aclanthology.org/D13-1170/) · [SST-2 dataset used here](https://huggingface.co/datasets/stanfordnlp/sst2)
