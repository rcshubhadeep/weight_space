# Weight-space workbench

[Research report: Two tasks, one gradient-free weight edit](docs/GRADIENT_FREE_COMPOSITION.md)

Experiments in specifying a behavior change, computing a small edit to a neural network, and testing what survives the edit. The current language-model experiments use LFM3-1.7B and a supervised, gradient-free rank-one solve. Applying the branch requires no external model or labeling service.

## Results at a glance

| Experiment | Result | Scope |
|---|---|---|
| BoolQ, counterbalanced A/B | 57.0% → **71.6%**, versus 64.1% for the best of 32 equal-norm random patches | All 3,270 development questions, each scored in both mappings |
| BoolQ, direct Yes/No probe | 80.7% → **80.9%**, one extra correct answer | Same A/B-fitted patch, 512 development questions |
| Field disambiguation | 79.8% → **82.9%** on unseen document types | 299 documents from a public extraction dataset, adapted to two choices |
| Synthetic urgency | 53.1% → **78.1%** on novel phrases, with sentiment at 98.4% | Direct write to one LFM3 matrix with exact restoration |
| Deletion-mask composition | **2 of 18** individually successful pairs failed when combined | Two disjoint-mask counterexamples in a 48-seed toy study |
| LFM3 patch composition | **0 of 160** fitted anchor choices changed with both cross-protected patches active | On held-out prompts, 207 of 320 choices passed the real-arithmetic margin check. None of those flipped in FP16 |
| Algebraic decision intervals | **280 of 320** held-out choices proved stable through the full strength interval | Both cross-protected patches active. Exact arithmetic over captured tensors |

BoolQ's clearest gain is answer-mapping robustness. The direct Yes/No result provides little evidence of general reading-comprehension improvement. See the [results and caveats](docs/RESULTS.md) for each experiment, including random controls, preservation failures, and the earlier distillation work.

## Reproduce

- [Environment, pinned downloads, and commands](docs/REPLICATION.md)
- [Mathematical derivation and executable checks](docs/METHOD.md)
- [Comparison with weight-space-lp, ROME, and related work](docs/RELATED_WORK.md)
- [Frozen reports and evidence manifest](results/README.md)

With a PyTorch environment active, run the small tests and verify the frozen results:

```sh
python -m unittest discover -s tests -v
python scripts/verify_evidence.py
```

## Layout

| Path | Contents |
|---|---|
| [weight_space/](weight_space/) | Evaluation helpers and rank-one numerical routines |
| [examples/](examples/) | Runnable experiments, from toy models to LFM3 benchmarks |
| [notebooks/](notebooks/) | Exploratory notebooks |
| [docs/](docs/) | Methods, replication, findings, and provenance |
| [results/](results/) | Frozen JSON reports and input fingerprints |
| `artifacts/` | Local datasets, patches, checkpoints, and new run outputs |

[Research agenda](RESEARCH.md)

[Compile standalone classifiers from LFM token embeddings and try the word-order demo](docs/REPLICATION.md#standalone-embedding-children).

[Order-sensitive children, LP readout, reversal demo and failure cases](docs/ORDER_INHERITANCE.md).

[Recovering a rank-one program from reversals: theorem, proof, compiler, controls, and failures](docs/REVERSAL_RECOVERY.md).

[Experimental record by research question, intervention, outcome, and limitation](docs/EXPERIMENTS.md).

[SST-2: improving the full Liquid model and benchmarking standalone children](docs/SST2.md).

[Composing the BoolQ and SST-2 edits into one always-active LFM patch](docs/BOOLQ_SST2_COMPOSITION.md).
