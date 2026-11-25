# Inheriting word order from LFM3

The task is to report the sentiment of the service actually experienced. Each pair swaps two adjectives and reverses the correct answer. Each frame puts the actual event first or second. The 36 fitting sentences cross three adjectives per polarity and two frames. The 64 evaluation sentences cross four new adjectives per polarity and two new frames. The stimuli reuse words and frames extensively.

## First-layer extraction

The compiler folds the input embedding, first operator RMS normalization and three input projections into per-token tables for the gates B*x and C. Each selected channel retains its seven-tap causal convolution. We mean-pool its gated response across positions. This recreates a subset of the first convolution's internal features in float32. A supervised readout selects channels by absolute training correlation, centers and scales features on the fit set, clips to [-1,1], and includes an intercept. A simplex solve maximizes the minimum signed training margin under a unit L1 budget on coefficients including the intercept. The reported primal/dual gap is a numerical check for the fixed features. The channel search itself is heuristic.

The 16-channel child classified 36/36 fitting sentences and 23/64 evaluation sentences. Ablating the six historical convolution taps gave 18/36 and 32/64. The width-matched vocabulary permutation gave 36/36 and 28/64. The exported child agreed with compilation on all 100 decisions. Keeping the fit adjectives and changing only the frames dropped the inherited child to 5/36. Keeping the fit frames and changing only the adjectives gave 46/64. Parent comparison uses a single negative=A, positive=B mapping and a task instruction.

## A factored alternative

After inspecting this failure we designed a second compiler for the same task. It has explicit supervision about roles. One embedding direction contrasts received with expected/expecting, the other contrasts the three positive fit adjectives with the three negative ones. Inner products with centered normalized LFM vocabulary embeddings give each token two numbers. English words are lowercased and represented by the average of their subtoken scores. The child sums products of a word's role score and the next word's sentiment score. This is a rank-one bigram scoring matrix. The adjacency rule and seeds come from the experimenter. Compilation uses no teacher predictions or gradient updates.

There is an exact prediction about swapping adjectives. If the two slots have preceding words u,v, and both are followed by the same noun, swapping adjectives a,b changes the score (original minus swapped) by (role(u)-role(v)) times (sentiment(a)-sentiment(b)). All other contributions cancel. This predicts score ordering. Correct decisions on both members also require the two scores to straddle zero.

The 1,000,344-byte float32 table shares the tokenizer. It scored 64/64 on the reused evaluation suite. The same-size control with one fixed vocabulary permutation and recompiled axes scored 26/64. Adding adverbs between the cues and adjectives reduced accuracy to 57/64. Inserting “anything but” before the actual outcome and reversing its label reduced it to 13/64. These templates and their labels operationalize a binary polarity task. This is an exploratory demonstration of a human-specified composition rule. An independent corpus and additional permutation seeds are needed for broader claims.

## Replication

Run from the repository root with the research requirements and pinned model in the existing GPU environment.

```powershell
$model = "$env:USERPROFILE/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
$env:HF_HUB_OFFLINE = '1'
$env:HF_MODULES_CACHE = Join-Path (Get-Location) '.venv/hf-modules'
python -m examples.convolution_child --compile $model
python -m examples.convolution_child_audit $model
python -m examples.factored_child --compile $model
python -m examples.factored_child --message "I predicted horrible service but experienced wonderful service."
python -m examples.factored_child --message "I predicted wonderful service but experienced horrible service."
python scripts/verify_evidence.py
```

[Compiler results](../results/2026-11-25/convolution_child.json) · [Crossed parent comparison](../results/2026-11-25/convolution_child_crossed.json) · [Factored results and stress tests](../results/2026-11-25/factored_child.json)
