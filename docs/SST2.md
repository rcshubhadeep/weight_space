# SST-2 standalone sentiment benchmark

Compare four systems on the 872 labeled validation examples from stanfordnlp/sst2. The existing reversal child stays frozen. The new compiler gets SST-2 supervision. The lexical system gets the same fitting labels. The full LFM3 parent gets an instruction with a fixed negative=A, positive=B mapping. The parent receives no fitting examples.

## Protocol

Shuffle the official training rows with NumPy seed 7331. Keep the first occurrence of each lowercase, whitespace-normalized string, excluding exact normalized validation strings. Reserve the first 1,024 retained rows for calibration and fit on the remainder. The official training split contains phrases. Distinct strings may share source material, especially between fit and calibration. Validation labels are used only for final scoring.

Compare the previous compiled child against a token unigram/bigram class-conditional log-frequency baseline and a supervised, closed-form ridge readout of mean frozen LFM input embeddings. Export the ridge direction as one scalar per vocabulary item. Calibration chooses the score threshold, ridge penalty from 0.01, 0.1, 1 and lexical mixture from 0, 0.25, 0.5, 0.75, 1 by accuracy. The lexical baseline gets its own threshold. Features and mixture scales use fitting rows only. Lexical and compiled inference include tokenization. No parent prediction is used to compile the child.

Report warm batch-one median end-to-end latency on the first 64 validation texts. The standalone classifiers execute on the host and the full parent executes on CUDA. Parent latency includes prompt formatting and transferring its result to the host. Size includes the tokenizer and serialized classifier weights or parent checkpoint. Timing excludes startup and loading weights.

## Replication

From the project environment after compiling the historical reversal child:

```powershell
.\.venv\Scripts\hf.exe download stanfordnlp/sst2 data/train-00000-of-00001.parquet data/validation-00000-of-00001.parquet --repo-type dataset --revision 8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb --local-dir artifacts/sst2
$model = "$env:USERPROFILE/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
$env:HF_MODULES_CACHE = Join-Path (Get-Location) '.venv/hf-modules'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m examples.sst2_benchmark --model $model
```

The run saves input file hashes, calibration trials, paired validation counts, serialized models and timing measurements in artifacts/sst2/benchmark.

## Results for the standalone child

The fit used 65,954 distinct training strings and 1,024 calibration strings. Calibration selected ridge 1 and a 75% lexical mixture on standardized scores. On 872 reviews, the frozen reversal child scored 422, the trained lexical baseline 706, the compiled mixture 717, and the fixed-A/B full parent 760. The mixture fixed 35 lexical errors and broke 24 correct lexical decisions. Its median warm batch-one time was 0.157 ms on the host versus 66.8 ms for the parent on CUDA. Its serialized footprint with tokenizer was 21,355,506 bytes. This comparison uses one split and one machine.

## Editing the full Liquid model

Run the edit benchmark in the same environment after the dataset download:

```powershell
.\.venv\Scripts\python.exe -m examples.lfm_sst2_edit --model $model
.\.venv\Scripts\python.exe -m examples.lfm_sst2_edit --model $model --replay
```

Use seed 1021 to sample 512 fit and 256 calibration strings without duplicate normalized text or exact validation matches. Fit a final-feed-forward rank-one branch separately for fixed A/B and direct negative/positive answer tokens. Choose its strength by calibration accuracy from 0, 0.125, 0.25, 0.5, 1, 2, breaking ties toward smaller edits. Compare live model forwards with and without the branch and compare a training-calibrated threshold on unedited logits. The binary scoring protocol restricts each prediction to two tokens. Each branch uses label supervision and a closed-form solve. No gradient updates or teacher predictions enter that solve. The original W2 digest is checked before and after evaluation.

### Full-head replay

Reload the serialized branches and score all 872 reviews through the ordinary vocabulary head at batch size 12. The fixed-A/B base reproduces the independent parent benchmark. The initial two-row readout at batch size 16 has slightly different FP16 decision counts. Use paired base and edited scores from the same execution path.

| Answer tokens | Base | Edited | Threshold-only | Fixed | Broken |
|---|---:|---:|---:|---:|---:|
| Fixed A/B | 760 | 785 | 759 | 53 | 28 |
| Negative/positive | 735 | 763 | 783 | 77 | 49 |

Each serialized patch occupies 39,124 bytes. The two protocols use separately fitted patches. The validation gains concern binary sentiment under these prompts. Cross-task retention and transfer to new datasets remain open.

[Frozen full-model results](../results/2026-11-25/sst2_liquid_edit.json) and [standalone results](../results/2026-11-25/sst2_child.json).
