# Weight-space workbench

An early CPU-friendly Python workbench for exploring the ideas in [RESEARCH.md](RESEARCH.md). The first example deliberately breaks one weight of a tiny paddle policy, tests a manual repair, and records what happened.

```bash
uv sync
uv run python -m examples.paddle_repair
uv run python -m unittest discover -s tests
```

The example saves original and edited weights as safetensors and a detailed JSON report in `artifacts/paddle_repair/`. Rerunning it overwrites those files. The model is `torch.nn.Linear(2, 2)` with inputs `[paddle_position, ball_position]` and scores `[left, right]`; to reload a saved state, construct that architecture and use `safetensors.torch.load_file` with `model.load_state_dict(...)`.

Read [examples/paddle_repair.py](examples/paddle_repair.py) to see the specification, edit, and comparison. A tie in the action scores selects the first action; this example intentionally excludes aligned positions because it has no stay action. Its four cases demonstrate the workflow, not generalization or a validated repair method. No optimizer or LP solver is used yet.

## Trained network: spurious correlation

```bash
uv run python -m examples.spurious_feature
```

This trains a small classifier on synthetic data where a misleading feature usually agrees with the label. It then uses labeled calibration examples where that feature conflicts with the label to find a hidden-activation direction. A rank-one output-weight edit and ten random-direction controls each select their edit strength using a separate validation set. The report evaluates them on a separate balanced test set, broken down by whether the misleading feature agrees or conflicts with the label. The original and edited weights and report are written to `artifacts/spurious_feature/`; rerunning overwrites them.

Training uses gradients; the subsequent direction construction and edits do not. The data generator supplies the conflict labels, so this does not solve the harder problem of discovering the failure without that information. The controls tune their strengths independently, so their edit norms can differ. Results are for one seed and are not evidence of general effectiveness.
