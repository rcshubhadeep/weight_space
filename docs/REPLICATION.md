# Replication

Run these commands from the repository root. The [manifest](../results/2026-11-25/manifest.json) records the model and dataset revisions and their SHA-256 hashes. New runs write to `artifacts/`.

## Environment

The archived language-model runs used Windows x86-64, Python 3.12.15, PyTorch 2.15.0+cu134, Transformers 5.22.1, and an RTX 2080 Super Max-Q with 8 GiB VRAM and driver 621.24. Inference used CUDA float16 and SDPA. [Pinned direct dependencies](../requirements/research.txt) describe the environment used for the runs. CPU mode uses float32. Hardware and library changes may affect borderline decisions.

```powershell
uv venv --python 3.12 artifacts/replication-venv
$py = ".\artifacts\replication-venv\Scripts\python.exe"
$hf = ".\artifacts\replication-venv\Scripts\hf.exe"
uv --no-config pip install --python $py --index https://download.pytorch.org/whl/cu134 "torch==2.15.0+cu134"
uv --no-config pip install --python $py -r requirements/research.txt
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Use the dedicated environment above for the CUDA runs. The repository's separate `uv.lock` is CPU-oriented. An existing compatible environment can use its own Python and `hf` paths. For Linux, use the virtual environment's `bin/python` and `bin/hf` with equivalent shell syntax.

## Download and verify the inputs

```powershell
$modelRev = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
$extractionRev = "58f0d07d46cd58fcfedc836c0011653872eb10a9"
$boolqRev = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
$model = "artifacts/inputs/lfm3/$modelRev"
$extraction = "artifacts/inputs/extraction/$extractionRev"
$boolq = "artifacts/boolq/data"
& $hf download LiquidAI/LFM3-1.7B --revision $modelRev --local-dir $model
& $hf download LiquidAI/structured-extraction train.parquet test.parquet test_unseen.parquet --repo-type dataset --revision $extractionRev --local-dir $extraction
& $hf download google/boolq data/train-00000-of-00001.parquet data/validation-00000-of-00001.parquet --repo-type dataset --revision $boolqRev --local-dir artifacts/boolq
& $py scripts/verify_evidence.py --model-dir $model --extraction-dir $extraction --boolq-dir $boolq
$env:HF_HUB_OFFLINE = '1'
$env:HF_MODULES_CACHE = Join-Path (Get-Location) 'artifacts/hf-modules'
```

The loader executes the checkpoint's architecture code. Use the pinned upstream snapshot or a trusted local copy. Keep the revision as the model directory name for direct-write replay.

## Run the experiments

```powershell
& $py -m unittest discover -s tests -v
& $py -m examples.paddle_repair
& $py -m examples.spurious_feature
& $py -m examples.deletion_composition --seeds 48
& $py -m examples.family run --teacher $model
& $py -m examples.lfm_weight_edit --model $model
& $py -m examples.lfm_weight_edit --model $model --patch artifacts/lfm_weight_edit/patch.safetensors |
    Set-Content -Encoding utf8 artifacts/lfm_weight_edit/replay.json
& $py scripts/verify_evidence.py --replay artifacts/lfm_weight_edit/replay.json
& $py -m examples.lfm_extraction_benchmark --model $model --data $extraction
& $py -m examples.lfm_boolq_benchmark --model $model --data $boolq --random-count 32
& $py -m examples.lfm_boolq_benchmark --model $model --data $boolq --native-probe
& $py -m examples.lfm_patch_composition --model $model --data $extraction
& $py -m examples.lfm_patch_composition --model $model --data $extraction --algebraic
```

The direct Yes/No probe appends to the report from the full BoolQ run. The strict replay check compares exact metrics from the recorded precision. Compare report values and record any differences when using another environment. The original patch hash documents provenance and regenerated patch files can be compared by their behavior and metadata.

The algebraic run writes `artifacts/lfm_patch_composition/algebraic_report.json`.
It uses Python rational arithmetic over the cached model activations and verifies
the symbolic reduction with SymPy in the unit tests.
## Standalone embedding children

The GPU compiler reads LFM3's input embedding matrix. For each task, it averages normalized token embeddings within each labeled phrase, then subtracts the negative-class mean from the positive-class mean. Projecting every vocabulary row onto this direction produces one signed byte per token after quantization. The overlap control uses the same phrase weights and calibration with an identity similarity matrix. Each child averages its token scores and compares the result to a threshold chosen on 32 calibration messages. Compilation uses no teacher predictions or gradient updates.

In the project environment, run:

```powershell
$model = "$env:USERPROFILE/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
python -m examples.embedding_children --compile $model
python -m examples.embedding_children --message "I expected terrible service and got excellent service."
python -m examples.embedding_children --message "I expected excellent service and got terrible service."
```

Inference reads the exported tables and shared tokenizer. The paired demo yields the same sentiment margin on both messages because the token counts agree. On 128 synthetic messages, urgency scored 78 correct and sentiment scored 64. The overlap control scored 72 on each. The evaluation crosses eight novel phrases per task with two templates. The 128 messages reuse phrases, limiting generalization claims. See the [frozen report](../results/2026-11-25/lfm_embedding_children.json).
