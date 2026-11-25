"""Benchmark an automatically solved, removable LFM3 patch on field disambiguation."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import unicodedata

import torch
from torch.nn import functional as F

from examples.lfm_weight_edit import Cases, Experiment, classification, general_cases, retention


OUTPUT = Path("artifacts/lfm_extraction_benchmark")


@dataclass(frozen=True)
class Example:
    document: str
    field: str
    description: str
    correct: str
    distractor: str
    language: str
    document_type: str


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def eligible_values(row):
    document = normalize(row["document"])
    schema = json.loads(row["schema"]).get("properties", {})
    groups = defaultdict(list)
    for name, value in json.loads(row["extraction"]).items():
        if type(value) not in (str, int, float) or name not in schema:
            continue
        literal = normalize(str(value))
        if isinstance(value, str) and (len(literal) < 4 or len(literal) > 100):
            continue
        if not literal or not re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", document):
            continue
        kind = ("number", None) if not isinstance(value, str) else ("text", schema[name].get("format"))
        groups[kind].append((name, str(value), literal))
    return schema, groups


def load_split(path, seed):
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    rows = pq.read_table(path).to_pylist()
    examples = []
    for row in rows:
        schema, groups = eligible_values(row)
        options = []
        for fields in groups.values():
            for name, correct, literal in fields:
                distractors = [value for other, value, normalized in fields
                               if other != name and literal not in normalized and normalized not in literal]
                if distractors:
                    options.append((name, correct, rng.choice(distractors)))
        if options:
            name, correct, distractor = rng.choice(options)
            examples.append(Example(
                row["document"], name, schema[name].get("description", ""),
                correct, distractor, row["language"], row["doc_type"],
            ))
    rng.shuffle(examples)
    return len(rows), examples


def make_cases(examples):
    prompts = []
    for item in examples:
        question = f"Which value belongs to the field {item.field}? {item.description}"
        for a, b in ((item.correct, item.distractor), (item.distractor, item.correct)):
            prompts.append(
                f"{question} Choose only A or B.\n"
                f"A = {json.dumps(a, ensure_ascii=False)}\n"
                f"B = {json.dumps(b, ensure_ascii=False)}\n\nDocument:\n{item.document}"
            )
    return Cases(prompts, torch.tensor([False, True] * len(examples)))


def correctness(cases, scores):
    return ((scores[:, 1] - scores[:, 0] > 0) == cases.expected_b).view(-1, 2)


def statistics(cases, before, after):
    base, edit = correctness(cases, before), correctness(cases, after)
    delta = (edit.float() - base.float()).mean(dim=1)
    generator = torch.Generator().manual_seed(725)
    indices = torch.randint(len(delta), (2000, len(delta)), generator=generator)
    ci = delta[indices].mean(dim=1).quantile(torch.tensor([0.025, 0.975]))
    return {
        "documents": len(base),
        "base_accuracy": float(base.float().mean()),
        "patched_accuracy": float(edit.float().mean()),
        "base_both_orders_correct": float(base.all(dim=1).float().mean()),
        "patched_both_orders_correct": float(edit.all(dim=1).float().mean()),
        "paired_delta_95_percent_bootstrap_interval": ci.tolist(),
    }


def audit(experiment, cases, key, scale, predicted, count=48):
    subset = Cases(cases.prompts[:count], cases.expected_b[:count])
    with experiment.attached(key, scale):
        live, _ = experiment.forward(subset, answers_only=True)
    expected = predicted[:len(live)]
    return {
        "prompts": len(live),
        "max_absolute_logit_difference": float((live - expected).abs().max()),
        "agreement": float((correctness(subset, live) == correctness(subset, expected)).float().mean()),
    }


def run(model_path, data_path, device):
    source = {}
    for split, seed in (("train", 831), ("test", 832), ("test_unseen", 833)):
        total, examples = load_split(data_path / f"{split}.parquet", seed)
        source[split] = {
            "source_documents": total, "eligible_documents": len(examples),
            "languages": dict(Counter(item.language for item in examples)),
        }
        if split == "train":
            if len(examples) < 192:
                raise ValueError("Need at least 192 eligible training documents")
            fit, validation = make_cases(examples[:96]), make_cases(examples[96:192])
            held_out = {}
        else:
            if not examples:
                raise ValueError(f"No eligible documents for {split}")
            held_out[split] = make_cases(examples)
    print(json.dumps(source), flush=True)

    experiment = Experiment(model_path, device)
    original_digest = experiment.weight_digest()
    fit_base, fit_capture = experiment.forward(fit, capture=True, answers_only=True)
    preserve = [classification(0, task, count=16) for task in ("sentiment", "urgency")]
    protection = torch.cat([
        experiment.forward(cases, capture=True, answers_only=True)[1][0] for cases in preserve
    ])
    key = experiment.solve(fit, fit_base, fit_capture, protection)
    unguarded = F.normalize(experiment.solve(fit, fit_base, fit_capture), dim=0) * key.norm()
    generator = torch.Generator().manual_seed(613)
    variants = {"constrained": key, "unconstrained_norm_matched": unguarded}
    for i in range(3):
        random_key = F.normalize(torch.randn(key.numel(), generator=generator), dim=0)
        variants[f"random_{i + 1}"] = random_key.to(device) * key.norm()

    valid_base, valid_capture = experiment.forward(validation, capture=True, answers_only=True)
    trials = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        scores = experiment.score_cached_answers(valid_capture, key, scale)
        trials.append({"scale": scale, "accuracy": float(correctness(validation, scores).float().mean())})
    scale = max(trials, key=lambda trial: (trial["accuracy"], -trial["scale"]))["scale"]
    base_accuracy = float(correctness(validation, valid_base).float().mean())
    print(f"Validation base {base_accuracy:.2%}; scale sweep {trials}; selected {scale}", flush=True)
    # If validation selects no edit, still report the unit-scale variants as diagnostic controls.
    control_scale = scale if scale else 1.0

    tests = {}
    for split, cases in held_out.items():
        print(f"Running LFM3 on {split}: {len(cases.prompts)} prompts", flush=True)
        before, capture = experiment.forward(cases, capture=True, answers_only=True)
        outcomes = {}
        for name, branch in variants.items():
            strength = scale if name == "constrained" else control_scale
            after = experiment.score_cached_answers(capture, branch, strength)
            outcomes[name] = statistics(cases, before, after)
            if name == "constrained":
                outcomes[name]["live_forward_audit"] = audit(experiment, cases, key, scale, after)
        tests[split] = outcomes
        print(split, json.dumps(outcomes), flush=True)

    preserved = {}
    controls = {task: classification(2, task, ("novel_phrases", "novel_phrases"))
                for task in ("sentiment", "urgency")}
    controls["general_next_token"] = general_cases(True)
    for name, cases in controls.items():
        before, _ = experiment.forward(cases)
        with experiment.attached(key, scale):
            after, _ = experiment.forward(cases)
        result = retention(before, after)
        if cases.expected_b is not None:
            result.update(base_accuracy=experiment.accuracy(cases, before),
                          patched_accuracy=experiment.accuracy(cases, after))
        preserved[name] = result
    if experiment.weight_digest() != original_digest:
        raise RuntimeError("The base weight matrix changed during the benchmark")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "LiquidAI/structured-extraction",
        "dataset_revision": data_path.name,
        "model_revision": model_path.name,
        "source_splits": source,
        "fit_documents": 96,
        "validation_documents": 96,
        "validation_base_accuracy": base_accuracy,
        "validation_trials": trials,
        "selected_scale": scale,
        "control_scale": control_scale,
        "held_out": tests,
        "preservation": preserved,
        "base_matrix_unchanged": True,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Preservation:", json.dumps(preserved), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted local LFM3-1.7B snapshot")
    parser.add_argument("--data", required=True, type=Path, help="Local structured-extraction snapshot")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.model, args.data, args.device)


if __name__ == "__main__":
    main()
