"""A deliberately small workbench for classification policies."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Case:
    """One input, expected action index, and the reason we test it."""

    name: str
    inputs: tuple[float, ...]
    expected: int
    group: str


def evaluate(model: nn.Module, cases: list[Case]) -> list[dict]:
    """Evaluate action scores; ties follow PyTorch argmax's first-index rule."""
    results = []
    for case in cases:
        with torch.no_grad():
            scores = model(torch.tensor([case.inputs], dtype=torch.float32))[0]
        action = int(scores.argmax().item())
        results.append({
            "name": case.name, "group": case.group, "inputs": list(case.inputs),
            "expected": case.expected, "actual": action,
            "scores": scores.tolist(), "passed": action == case.expected,
        })
    return results


def set_parameter(model: nn.Module, name: str, index: tuple[int, ...], value: float) -> dict:
    """Change one named parameter and return an audit record of the edit."""
    parameters = dict(model.named_parameters())
    if name not in parameters:
        raise ValueError(f"Unknown parameter {name!r}; available: {list(parameters)}")
    parameter = parameters[name]
    if len(index) != parameter.ndim or any(i < 0 or i >= n for i, n in zip(index, parameter.shape)):
        raise IndexError(f"Index {index} is invalid for {name} with shape {tuple(parameter.shape)}")
    with torch.no_grad():
        old = parameter[index].item()
        parameter[index] = value
        new = parameter[index].item()
    return {"parameter": name, "index": list(index), "before": old, "after": new}


def compare(before: list[dict], after: list[dict]) -> dict:
    """Summarize per-case changes and pass counts for each behavior group."""
    if len(before) != len(after):
        raise ValueError("Before and after must contain the same cases")
    groups = {}
    changes = []
    for old, new in zip(before, after):
        if any(old[key] != new[key] for key in ("name", "group", "inputs", "expected")):
            raise ValueError("Before and after must contain the same cases in the same order")
        group = groups.setdefault(old["group"], {"total": 0, "before_passed": 0, "after_passed": 0})
        group["total"] += 1
        group["before_passed"] += int(old["passed"])
        group["after_passed"] += int(new["passed"])
        changes.append({"name": old["name"], "before": old["actual"], "after": new["actual"],
                        "before_passed": old["passed"], "after_passed": new["passed"]})
    return {"groups": groups, "cases": changes}
