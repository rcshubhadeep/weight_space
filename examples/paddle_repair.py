"""Run with: uv run python -m examples.paddle_repair"""

import copy
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from weight_space import Case, compare, evaluate, set_parameter


def main():
    # Scores: [left, right]. One incorrect sign makes some left cases fail.
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[-1.0, -1.0], [-1.0, 1.0]])
        model.bias.zero_()

    cases = [
        Case("ball left", (5.0, 2.0), 0, "target"),
        Case("ball far left", (8.0, 1.0), 0, "target"),
        Case("ball right", (5.0, 8.0), 1, "preserve"),
        Case("ball far right", (1.0, 8.0), 1, "preserve"),
    ]
    before = evaluate(model, cases)
    edited = copy.deepcopy(model)
    edit = set_parameter(edited, "weight", (0, 0), 1.0)
    after = evaluate(edited, cases)
    report = {"edit": edit, "before": before, "after": after, "comparison": compare(before, after)}

    output = Path("artifacts/paddle_repair")
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(output / "before.safetensors"))
    save_file({k: v.detach().contiguous() for k, v in edited.state_dict().items()}, str(output / "after.safetensors"))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2))
    print(f"Saved weights and detailed report to {output}/")


if __name__ == "__main__":
    main()
