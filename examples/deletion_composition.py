"""Test whether two independently chosen deletion masks compose."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from examples.spurious_feature import data


GROUPS = ("a_conflict", "a_agree", "b_conflict", "b_agree")
TOLERANCE = 0.02
REQUIRED_GAIN = 0.05


def dataset(count, agreement, seed):
    features, labels, groups = [], [], []
    for task in range(2):
        x, y, agree = data(count, agreement, seed + task)
        task_id = x.new_full((count,), 2 * task - 1)
        absent = torch.zeros_like(task_id)
        shortcut_a, shortcut_b = (absent, x[:, 1]) if task else (x[:, 1], absent)
        features.append(torch.stack((x[:, 0], shortcut_a, shortcut_b, task_id), dim=-1))
        labels.append(y)
        groups.append(2 * task + agree.long())
    return torch.cat(features), torch.cat(labels), torch.cat(groups)


def train(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y, _ = dataset(2000, 0.97, seed * 10)
    for _ in range(200):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y.float()).backward()
        optimizer.step()
    return model.eval()


class Evaluator:
    def __init__(self, model, cases):
        self.x, self.y, group = cases
        self.membership = F.one_hot(group, len(GROUPS)).float()
        self.group_sizes = self.membership.sum(dim=0)
        self.hidden_size, self.input_size = model[0].weight.shape
        self.split = model[0].weight.numel()
        self.weights = torch.cat((model[0].weight.flatten(), model[2].weight.flatten())).detach()
        self.hidden_bias = model[0].bias.detach()
        self.output_bias = model[2].bias.detach()

    @torch.no_grad()
    def evaluate(self, masks):
        weights = self.weights.masked_fill(masks, 0)
        hidden_weights = weights[:, :self.split].reshape(-1, self.hidden_size, self.input_size)
        hidden = (self.x @ hidden_weights.transpose(-2, -1) + self.hidden_bias).relu()
        logits = (hidden @ weights[:, self.split:].unsqueeze(-1)).squeeze(-1) + self.output_bias
        accuracy = (((logits >= 0) == self.y).float() @ self.membership) / self.group_sizes
        loss = (F.softplus(logits * (1 - 2 * self.y)) @ self.membership) / self.group_sizes
        return accuracy, loss

    def empty(self):
        return torch.zeros_like(self.weights, dtype=torch.bool)


def search(evaluator, target, budget, allow_early_stop):
    mask = evaluator.empty()
    base_acc, _ = evaluator.evaluate(mask.unsqueeze(0))
    preserve = [i for i in range(len(GROUPS)) if i not in target]

    def score(masks):
        acc, loss = evaluator.evaluate(masks)
        damage = (base_acc[:, preserve] - acc[:, preserve] - TOLERANCE).clamp_min(0)
        return loss[:, target].mean(dim=1) + 20 * damage.sum(dim=1)

    current = score(mask.unsqueeze(0))[0]
    for _ in range(budget):
        remaining = (~mask).nonzero().flatten()
        candidates = mask.repeat(len(remaining), 1)
        candidates[torch.arange(len(remaining)), remaining] = True
        scores = score(candidates)
        best = int(scores.argmin())
        if allow_early_stop and scores[best] >= current:
            break
        mask, current = candidates[best], scores[best]
    return mask


def random_controls(evaluator, union, seed, count=20):
    generator = torch.Generator().manual_seed(seed)
    masks = torch.zeros(count, union.numel(), dtype=torch.bool)
    for start, end in ((0, evaluator.split), (evaluator.split, union.numel())):
        budget = int(union[start:end].sum())
        for mask in masks:
            indices = torch.randperm(end - start, generator=generator)[:budget] + start
            mask[indices] = True
    return masks


def accuracy_record(scores):
    return dict(zip(GROUPS, scores.tolist()))


def passes(scores, base, target):
    return all(
        float(scores[i] - base[i]) >= (REQUIRED_GAIN if i in target else -TOLERANCE)
        for i in range(len(GROUPS))
    )


def experiment(seed):
    model = train(seed)
    calibration = Evaluator(model, dataset(768, 0.5, 10000 + seed * 10))
    test = Evaluator(model, dataset(2048, 0.5, 20000 + seed * 10))
    a = search(calibration, [0], budget=4, allow_early_stop=True)
    b = search(calibration, [2], budget=4, allow_early_stop=True)
    union = a | b
    joint = search(calibration, [0, 2], budget=int(union.sum()), allow_early_stop=False)
    named = {"base": test.empty(), "a": a, "b": b, "union": union, "joint": joint}
    scores, _ = test.evaluate(torch.stack(list(named.values())))
    base, score_a, score_b, score_union, score_joint = scores
    control_scores, _ = test.evaluate(random_controls(test, union, seed))
    return {
        "seed": seed,
        "overlap": int((a & b).sum()),
        "deletions": {key: mask.nonzero().flatten().tolist() for key, mask in named.items()},
        "test_accuracy": {key: accuracy_record(score) for key, score in zip(named, scores)},
        "random_test_accuracy": [accuracy_record(score) for score in control_scores],
        "passes": {
            "a": passes(score_a, base, [0]),
            "b": passes(score_b, base, [2]),
            "union": passes(score_union, base, [0, 2]),
            "joint": passes(score_joint, base, [0, 2]),
            "random_count": sum(passes(score, base, [0, 2]) for score in control_scores),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    torch.set_num_threads(1)
    results = [experiment(seed) for seed in range(41, 41 + args.seeds)]
    both = [run for run in results if run["passes"]["a"] and run["passes"]["b"]]
    failures = [run for run in both if not run["passes"]["union"]]
    summary = {
        "runs": len(results),
        "both_individual_pass": len(both),
        "union_pass_given_individual_pass": len(both) - len(failures),
        "union_failure_seeds_given_individual_pass": [run["seed"] for run in failures],
        "disjoint_union_failure_seeds": [run["seed"] for run in failures if not run["overlap"]],
        "joint_recovers_union_failure_seeds": [run["seed"] for run in failures if run["passes"]["joint"]],
        "joint_pass": sum(run["passes"]["joint"] for run in results),
        "random_pass": sum(run["passes"]["random_count"] for run in results),
        "random_runs": 20 * len(results),
    }
    output = Path("artifacts/deletion_composition")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "minimum_target_accuracy_gain": REQUIRED_GAIN,
        "max_preservation_accuracy_drop": TOLERANCE,
        "mask_coordinates": "Row-major weights of first layer followed by final layer. Biases stay frozen.",
        "summary": summary,
        "runs": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
