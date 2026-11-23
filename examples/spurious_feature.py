"""Train a biased classifier and investigate a gradient-free weight edit.

Run: uv run python -m examples.spurious_feature
"""

import copy
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn


SEED = 41


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        return self.output(self.features(x))


def data(n, agreement, seed):
    """Feature 0 is a noisy signal; feature 1 is a spurious correlate."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    sign = 2 * y.float() - 1
    agree = torch.rand(n, generator=g) < agreement
    spur_sign = sign * torch.where(agree, 1.0, -1.0)
    x = torch.stack((sign + 0.6 * torch.randn(n, generator=g),
                     1.5 * spur_sign + 0.3 * torch.randn(n, generator=g)), dim=1)
    return x, y, agree


def accuracy(model, dataset):
    x, y, agree = dataset
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    result = {}
    for name, mask in (("overall", torch.ones_like(agree)), ("agree", agree), ("conflict", ~agree)):
        result[name] = {"n": int(mask.sum()), "accuracy": float((predictions[mask] == y[mask]).float().mean()) if mask.any() else None}
    return result


def edit(model, direction, strength):
    edited = copy.deepcopy(model)
    with torch.no_grad():
        # Change the class-1 versus class-0 logit difference by strength * (h · direction).
        edited.output.weight[1].add_(direction, alpha=strength / 2)
        edited.output.weight[0].add_(direction, alpha=-strength / 2)
    return edited


def select_strength(model, direction, validation):
    # Choose using validation only; tie-break toward the smaller edit.
    candidates = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    scores = [(s, accuracy(edit(model, direction, s), validation)["overall"]["accuracy"]) for s in candidates]
    best = max(scores, key=lambda item: (item[1], -item[0]))
    return best[0], [{"strength": s, "accuracy": a} for s, a in scores]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    train = data(4000, 0.95, 1)
    calibration = data(800, 0.5, 2)
    validation = data(800, 0.5, 3)
    test = data(4000, 0.5, 4)
    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    x, y, _ = train
    for _ in range(250):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    # Only the training above uses gradients. All subsequent edits are gradient-free.
    # Contrast class means in the conflict subset, where the spurious signal is misleading.
    x_cal, y_cal, agree_cal = calibration
    with torch.no_grad():
        hidden = model.features(x_cal)
        conflict = ~agree_cal
        direction = hidden[conflict & (y_cal == 1)].mean(0) - hidden[conflict & (y_cal == 0)].mean(0)
        direction = direction / direction.norm()
    strength, sweep = select_strength(model, direction, validation)
    guided = edit(model, direction, strength)

    # Random directions are unit length; tune each strength on the same validation set.
    # Their final edit norms can therefore differ from the guided edit's norm.
    generator = torch.Generator().manual_seed(123)
    controls = []
    for _ in range(10):
        random_direction = torch.randn(direction.shape, generator=generator)
        random_direction /= random_direction.norm()
        random_strength, _ = select_strength(model, random_direction, validation)
        controls.append({"strength": random_strength, "test": accuracy(edit(model, random_direction, random_strength), test)})

    report = {
        "seed": SEED,
        "setup": "Training correlation 95%; independent calibration, validation and balanced held-out test sets.",
        "method": "Conflict-set class activation contrast; rank-one output-weight edit; validation selects strength.",
        "train_loss": float(loss.item()), "validation_sweep": sweep, "selected_strength": strength,
        "base_test": accuracy(model, test), "guided_test": accuracy(guided, test),
        "random_direction_controls": controls,
        "changed_weights": int((guided.output.weight != model.output.weight).sum()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    output = Path("artifacts/spurious_feature")
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(output / "before.safetensors"))
    save_file({k: v.detach().contiguous() for k, v in guided.state_dict().items()}, str(output / "after.safetensors"))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("selected_strength", "base_test", "guided_test", "changed_weights", "total_parameters")}, indent=2))
    print("Random-direction test overall accuracies:", [round(c["test"]["overall"]["accuracy"], 3) for c in controls])
    print(f"Saved models and report to {output}/")


if __name__ == "__main__":
    main()
