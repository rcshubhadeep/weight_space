# Composing the BoolQ and SST-2 patches

**Question.** Can an always-active final-layer patch improve the same frozen LFM3-1.7B checkpoint on both public validation splits? Both original tasks use the same A/B output direction v. If b and s are their individually calibrated input keys, the sum of the edits is exactly v (alpha b + beta s)^T in real arithmetic. The exported model therefore needs one rank-one branch and no prompt-time task selection.

## Protocol

Reconstruct the protected BoolQ key from its original 128 fitting questions and load the saved fixed-A/B SST-2 branch fitted on 512 training strings. On their original disjoint-from-validation calibration sets, compare 25 coefficient pairs on the grid 0, 0.25, 0.5, 0.75, 1 for each key. For each task, compute the candidate gain over base as a fraction of its own patch's calibration gain. Select the pair maximizing the lower fraction, breaking ties by sum of fractions then smaller sum of coefficients. This prespecifies how the two tasks trade off despite having different prompt counts and base accuracies.

Evaluate base, both individual edits, full-strength sum, and selected patch on all 3,270 BoolQ development questions in both mappings and all 872 SST-2 validation reviews. Replay the final layer from cached unedited activations using the FP16 suffix and two-row output head. Audit selection by attaching the patch to the live model on the first batch of the first and last chunks per task. Counts from the SST-2 full-vocabulary replay use a different evaluation path. Compare each variant with controls from this run.

## Replication

After completing the BoolQ and SST-2 experiments in the same project environment, run:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_MODULES_CACHE=Join-Path (Get-Location) '.venv/hf-modules'
$model="$env:USERPROFILE/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
.\.venv\Scripts\python.exe -m examples.lfm_boolq_sst2_composition --model $model
```

The run saves the combined key and output direction and checks the original edited-layer matrix digest. The validation splits were used in previous experiments. The coefficient search uses training-split calibration data only. These measurements concern two forced-choice tasks and leave free generation and unrelated capabilities unevaluated.

## Validation results

| Task | Base | BoolQ only | SST-2 only | Full sum | Selected |
|---|---:|---:|---:|---:|---:|
| boolq | 3729/6540 (57.0%) | 4682/6540 (71.6%) | 4109/6540 (62.8%) | 4136/6540 (63.2%) | 4254/6540 (65.0%) |
| sst2 | 762/872 (87.4%) | 719/872 (82.5%) | 783/872 (89.8%) | 726/872 (83.3%) | 780/872 (89.4%) |

Calibration selected 0.25 times the BoolQ key plus the full SST-2 key. This retained 525/953 of the BoolQ gain and 18/21 of the SST-2 gain. Full-strength addition lost 36 correct sentiment answers relative to base. Both BoolQ mappings were correct on 1,068 questions with the selected patch versus 601 at baseline. All 51 live audit choices agreed with the cached replay. [Frozen evidence](../results/2026-11-25/boolq_sst2_composition.json).
