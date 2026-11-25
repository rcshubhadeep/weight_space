# Results and limitations

All percentages below describe the recorded runs. The [frozen evidence](../results/README.md) contains the full reports, and the [replication guide](REPLICATION.md) lists the commands.

## BoolQ

The fit and calibration used 128 questions each from the training split, sampled with disjoint passages. A total of 1,434 training rows shared exact passages with development and were excluded before sampling. All 3,270 development questions were evaluated with both A/B mappings.

| Condition | Accuracy | Both mappings correct |
|---|---:|---:|
| Base LFM3 | 57.0% | 18.4% |
| Shuffled-label solve | 57.2% | 31.6% |
| Best of 32 equal-norm random directions | 64.1% | See report |
| Unconstrained solve | 62.9% | 32.9% |
| Protected solve | **71.6%** | **53.5%** |

With A meaning No, accuracy rose from 42.2% to 70.4%. With A meaning Yes, accuracy moved from 71.8% to 72.8%. The question-resampled paired bootstrap interval for the average gain was +13.2 to +15.9 points. It preserves answer pairs within a question. Some questions share passages, so this interval does not account for passage-level clustering.

The same branch gained one correct answer on a 512-question direct Yes/No probe, from 80.7% to 80.9%. The evidence supports improved A/B mapping under the measured prompt. Broad reading-comprehension improvement remains open. The cache matched 70 of 70 audited live choices. The base tensor hash matched after evaluation. [Full report](../results/2026-11-25/boolq.json)

## Field disambiguation

This experiment derives a two-choice task from Liquid AI's structured-extraction annotations. Both values occur in the source document. Each eligible document contributes one question scored in both mappings.

| Published split | Eligible documents | Base | Protected solve |
|---|---:|---:|---:|
| Familiar document types | 486 | 80.8% | 82.6% |
| Unseen document types | 299 | 79.8% | 82.9% |

One of three equal-norm random edits reached 82.8% on familiar types and 82.3% on unseen types. The comparison against random edits is therefore mixed. The task tests field discrimination within an annotated synthetic corpus and should be reported separately from open-ended extraction. The source and model share a publisher, and the experiment did not establish pretraining independence. [Full report](../results/2026-11-25/lfm_extraction_benchmark.json)

## LFM3 synthetic urgency

On familiar phrases in fresh templates, the merged write moved accuracy from 73.4% to 92.2%. On unfamiliar phrases it moved accuracy from 53.1% to 78.1%. The same-norm unconstrained edit reached 84.4% on unfamiliar urgency phrases while sentiment fell from 100% to 50%. With constraints, sentiment was 98.4%. On ten unrelated prompts the constrained edit changed one top next token. Each classification row contains 32 messages in two answer orders. Results cover one model and a small synthetic test. [Full report](../results/2026-11-25/lfm_weight_edit.json)

## Deletion-mask composition

Both independently searched masks passed individually in 18 of 48 toy runs. Their union failed the preservation rule in two of those runs, seeds 42 and 83. The masks shared zero weight coordinates. A joint search passed at the same deletion count in both cases. The failure threshold was a two-point drop in any preservation group, and both misses were close to that cutoff. This is a counterexample to disjointness guaranteeing behavioral composition under the specified criterion. [Full report](../results/2026-11-25/deletion_composition.json)

## LFM3 patch composition

Two separately fitted branches target urgency and field disambiguation at the same down-projection. The cross-protected solve includes the other task's fitted activations in its protected span. Calibration picked an extraction scale of zero for that solve, so a separately labeled diagnostic uses the best nonzero calibration scale (0.5). Both branches are active in the diagnostic.

| Held-out task | Prompts | Independent own → combined accuracy | Cross-protected nonzero probe own → combined accuracy |
|---|---:|---:|---:|
| Urgency, unfamiliar phrases | 128 | 93.0% → 93.0% | 87.5% → 87.5% |
| Extraction, unseen types | 192 | 80.2% → 55.2% | 80.7% → 76.6% |

With both cross-protected branches active, none of the 160 fitted anchor choices changed. On held-out data, the sufficient real-arithmetic margin check passed 126 of 128 urgency prompts and 81 of 192 extraction prompts. None of those 207 choices flipped in the FP16 evaluation. All 40 held-out flips in that diagnostic occurred in the remaining 113 prompts. The live audits matched all 96 checked choices across the three conditions. The base matrix hash was unchanged.

Protection changes the fitted keys and their norms, and variant strengths are chosen on calibration. The held-out comparison therefore measures these configured pipelines. It does not isolate an equal-norm causal effect of projection. The margin check is local to a fixed prompt, layer, and output position. See the [derivation](METHOD.md) and [full report](../results/2026-11-25/lfm_patch_composition.json).

## Algebraic decision intervals

For each held-out prompt, the algebraic procedure tests all strengths of the
second patch from zero to its selected full scale, with the first patch fixed.
The exact rational input comes from captured binary tensor values. The fully
active cross-protected diagnostic covers the same prompts as the earlier run.

| Condition | Task | Prompts | Earlier margin bound | Exact interval check |
|---|---|---:|---:|---:|
| Both independent branches | Urgency | 128 | 99 | 126 |
| Both independent branches | Extraction | 192 | 28 | 112 |
| Both cross-protected branches, nonzero diagnostic | Urgency | 128 | 126 | 128 |
| Both cross-protected branches, nonzero diagnostic | Extraction | 192 | 81 | 152 |

No FP16 endpoint changes were observed among the exact-interval passes in
these runs. For one cross-protected extraction prompt, the first tie lies
between 0.0328979 and 0.0329018 of the requested second-patch strength.
The exact calculation concerns fixed captured activations and one A/B logit
comparison. It does not certify FP16 arithmetic or generation over multiple
tokens. The unit tests illustrate why checking endpoints alone is insufficient
in general. The observed endpoint signs in this experiment did not hide an
interior tie.

Clipping each cross-protected extraction prompt below its first tie reduced
cached FP16 answer changes from 40 to 4 and retained 87.7% of the requested
strength on average. Checking the rounded candidate and backing off further
when needed eliminated those four remaining changes while retaining 87.4%
of the requested strength on average. All 192 extraction prompts retained a
nonzero second branch. This is a diagnostic using held-out activations.
[Full algebraic report](../results/2026-11-25/lfm_algebraic_boundaries.json)

## Earlier small-network experiments

- **Paddle.** One hand edit repaired two deliberately broken cases and preserved the two other cases. This is a minimal workbench check. [Full report](../results/2026-11-25/paddle_repair.json)
- **Spurious feature.** The guided rank-one edit gave 87.58% versus 87.48% overall for the base. Two of ten random directions scored higher. Control norms could differ. This run supplies little evidence of an advantage for the guided method. [Full report](../results/2026-11-25/spurious_feature.json)

## Family distillation

The sentiment child scored 99.0% on familiar phrases in new templates and 42.2% on unfamiliar phrases. The grandchild scored 97.9% and 42.2%, respectively. With only task-relevant words unfamiliar, the gate still routed 44 of 64 messages to the grandchild, and accuracy fell to 59.4% despite teacher fallback. The urgency gate never selected a descendant on the measured held-out sets. This was ordinary distillation on a synthetic benchmark. [Full report](../results/2026-11-25/family.json)
