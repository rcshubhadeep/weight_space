# Related work and attribution

The project started with the behavior-repair framing in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and the idea of replacing search with a numerical solve in [weight-space-lp](https://github.com/nathanrs/weight-space-lp). The implementation here is in Python and PyTorch. It has no dependency on the linked LP project or its released traces.

| Question | weight-space-lp | This workbench |
|---|---|---|
| Intervention | Rank-one contribution at MLP down-projections | Same mathematical form, applied to the final LFM3 down-projection |
| Direction of the contribution | Loop direction from a prior agent edit or statistics of looping and productive traces | Difference of the two answer-token unembedding rows |
| Input-dependent magnitude | Sparse linear program with an L1 objective, signed constraints, and slack | Ridge least squares after projecting away protected inputs |
| Signal for the solve | Looping versus productive activations | Desired signed answer margins converted to shifts using finite differences through the model's final normalization and readout |
| Evaluation | Loop suppression and downstream reasoning or instruction-following benchmarks | Forced-choice answer mapping, field disambiguation, and preservation probes |
| Deployment | Weight updates and a constant steering baseline | Reversible merged-write experiment and removable branches on a frozen checkpoint |

The linked work also reports an automated direction estimated from traces without an LLM, and a constant steering baseline that performs about as well as its LP result on MATH-500. Automation and rank one are therefore shared ideas. See the [repository README](https://github.com/nathanrs/weight-space-lp#how-it-works) and [write-up](https://github.com/nathanrs/weight-space-lp/blob/main/docs/WRITEUP.md).

[ROME](https://arxiv.org/abs/2202.05262) is established prior art for rank-one feed-forward edits. It uses causal tracing to choose middle-layer sites for editing factual associations. This project uses a fixed final block and fits a shared edit to examples of a task, with explicitly designated preservation inputs.

[Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885) studies models reasoning about their own edits. The LFM3 experiments here use an external numerical procedure with model activations and supplied labels. [Deletion Tuning](https://arxiv.org/abs/2611.10866) learns weight-deletion masks with gradients and raises mask composition as a future direction. Our toy composition experiment searches masks without gradients and records failures when independently chosen masks are combined.

The [Family pilot](RESULTS.md#family-distillation) is a knowledge-distillation experiment. Passing predictive behavior to smaller networks follows existing [distillation work](https://arxiv.org/abs/1503.02531). The [self-replication study](https://arxiv.org/abs/2611.25839) motivated the lineage and drift questions.

The contribution to describe at release is the **implemented and evaluated combination** of a protected ridge solve, answer-margin targets, and a counterbalanced protocol that detects when an edit is specific to its response format. The current evidence covers one checkpoint and the listed tasks. Priority and general effectiveness remain open.
