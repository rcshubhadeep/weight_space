"""Evaluate a locally solved, removable rank-one branch on the full BoolQ dev split."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from examples.lfm_extraction_benchmark import correctness, statistics
from examples.lfm_weight_edit import Cases, Experiment, classification, general_cases, retention


OUTPUT = Path("artifacts/boolq")
FIT_COUNT = 128
CALIBRATION_COUNT = 128
RANDOM_COUNT = 32


def make_cases(rows):
    prompts, labels = [], []
    for row in rows:
        for reverse in (False, True):
            a, b = ("Yes", "No") if reverse else ("No", "Yes")
            prompts.append(
                "Answer the yes/no question using the passage. Choose only A or B.\n"
                f"A = {a}; B = {b}.\n\nPassage: {row['passage']}\n\nQuestion: {row['question']}?"
            )
            labels.append(row["answer"] ^ reverse)
    return Cases(prompts, torch.tensor(labels))


def load_data(data):
    import pyarrow.parquet as pq

    train = pq.read_table(data / "train-00000-of-00001.parquet").to_pylist()
    test = pq.read_table(data / "validation-00000-of-00001.parquet").to_pylist()
    if (len(train), len(test)) != (9427, 3270):
        raise ValueError("Expected the full published BoolQ train and development splits")
    passages = {row["passage"] for row in test}
    exact_overlap = sum(row["passage"] in passages for row in train)
    sampled = list(train)
    random.Random(907).shuffle(sampled)
    independent = []
    for row in sampled:
        passage = row["passage"]
        if passage in passages:
            continue
        independent.append(row)
        passages.add(passage)
        if len(independent) == FIT_COUNT + CALIBRATION_COUNT:
            break
    if len(independent) != FIT_COUNT + CALIBRATION_COUNT:
        raise ValueError("Not enough distinct training passages after excluding overlap")
    return (
        sorted(independent[:FIT_COUNT], key=lambda row: len(row["passage"])),
        sorted(independent[FIT_COUNT:], key=lambda row: len(row["passage"])),
        sorted(test, key=lambda row: len(row["passage"])),
        exact_overlap,
    )


def batch_size(rows):
    # Long passages get smaller batches on the 8 GiB GPU.
    longest = max(len(row["passage"]) for row in rows)
    return max(2, min(16, 18000 // max(1, longest)))


def measure(cases, scores):
    correct = correctness(cases, scores)
    return {
        "accuracy": float(correct.float().mean()),
        "both_orders_correct": float(correct.all(dim=1).float().mean()),
        "no_is_a_accuracy": float(correct[:, 0].float().mean()),
        "yes_is_a_accuracy": float(correct[:, 1].float().mean()),
    }


def check_preservation(experiment, key, scale):
    suites = {
        task: classification(2, task, ("novel_phrases", "novel_phrases"))
        for task in ("sentiment", "urgency")
    }
    suites["general_next_token"] = general_cases(True)
    results = {}
    for name, cases in suites.items():
        before, _ = experiment.forward(cases)
        with experiment.attached(key, scale):
            after, _ = experiment.forward(cases)
        result = retention(before, after)
        if cases.expected_b is not None:
            result.update(base_accuracy=experiment.accuracy(cases, before),
                          patched_accuracy=experiment.accuracy(cases, after))
        results[name] = result
    return results


def fitting_inputs(experiment, fit_rows):
    fit = make_cases(fit_rows)
    fit_scores, fit_capture = experiment.forward(
        fit, capture=True, answers_only=True, batch_size=batch_size(fit_rows),
    )
    preservation_fit = [classification(0, task, count=16) for task in ("sentiment", "urgency")]
    protected_keys = torch.cat([
        experiment.forward(cases, capture=True, answers_only=True)[1][0] for cases in preservation_fit
    ])
    return fit, fit_scores, fit_capture, protected_keys


def run(model_path, data, device, random_count):
    fit_rows, calibration_rows, test_rows, exact_overlap = load_data(data)
    print(f"BoolQ: {len(fit_rows)} fit, {len(calibration_rows)} calibration, {len(test_rows)} held out", flush=True)
    experiment = Experiment(model_path, device)
    base_digest = experiment.weight_digest()
    fit, fit_scores, fit_capture, protected_keys = fitting_inputs(experiment, fit_rows)
    key = experiment.solve(fit, fit_scores, fit_capture, protected_keys)
    if not key.isfinite().all() or not key.norm() > 0:
        raise RuntimeError("The solve did not produce a nonzero finite key")

    generator = torch.Generator().manual_seed(908)
    order = torch.randperm(FIT_COUNT, generator=generator)
    shuffled_fit = Cases(fit.prompts, fit.expected_b.reshape(-1, 2)[order].flatten())
    shuffled_key = experiment.solve(shuffled_fit, fit_scores, fit_capture, protected_keys)
    unguarded_key = experiment.solve(fit, fit_scores, fit_capture)
    variants = {
        "solved": key,
        "shuffled_label": F.normalize(shuffled_key, dim=0) * key.norm(),
        "unconstrained": F.normalize(unguarded_key, dim=0) * key.norm(),
    }
    for i in range(1, random_count + 1):
        noise = torch.randn(key.numel(), generator=generator).to(device)
        variants[f"random_{i:02d}"] = F.normalize(noise, dim=0) * key.norm()
    del fit_capture, protected_keys

    calibration = make_cases(calibration_rows)
    cal_scores, cal_capture = experiment.forward(
        calibration, capture=True, answers_only=True, batch_size=batch_size(calibration_rows),
    )
    trials = [{"scale": 0.0, **measure(calibration, cal_scores)}]
    for scale in (0.125, 0.25, 0.5, 1.0, 2.0):
        scores = experiment.score_cached_answers(cal_capture, key, scale)
        trials.append({"scale": scale, **measure(calibration, scores)})
    selected = max(trials, key=lambda trial: (trial["accuracy"], -trial["scale"]))["scale"]
    # The best nonzero calibration setting is still informative if the deployment rule abstains.
    probe = max(trials[1:], key=lambda trial: (trial["accuracy"], -trial["scale"]))["scale"]
    del cal_capture
    print(f"Calibration {trials}; selected {selected}; experimental nonzero scale {probe}", flush=True)

    baseline_parts, target_parts = [], []
    variant_parts = {name: [] for name in variants}
    audits = []
    chunk_size = 256
    audit_chunks = {0, len(test_rows) // chunk_size // 2, (len(test_rows) - 1) // chunk_size}
    for start in range(0, len(test_rows), chunk_size):
        rows = test_rows[start:start + chunk_size]
        cases = make_cases(rows)
        size = batch_size(rows)
        base, capture = experiment.forward(cases, capture=True, answers_only=True, batch_size=size)
        baseline_parts.append(base)
        target_parts.append(cases.expected_b)
        for name, branch in variants.items():
            variant_parts[name].append(experiment.score_cached_answers(capture, branch, probe))
        if start // chunk_size in audit_chunks:
            count = min(size * 2, len(cases.prompts))
            subset = Cases(cases.prompts[:count], cases.expected_b[:count])
            with experiment.attached(key, probe):
                live, _ = experiment.forward(subset, answers_only=True, batch_size=size)
            cached = variant_parts["solved"][-1][:count]
            audits.append({
                "first_document": start, "prompts": len(live),
                "max_absolute_logit_difference": float((live - cached).abs().max()),
                "matching_choices": int(((live[:, 1] > live[:, 0]) == (cached[:, 1] > cached[:, 0])).sum()),
            })
        del capture
        print(f"Held out: {min(start + len(rows), len(test_rows))}/{len(test_rows)} questions", flush=True)

    labels = Cases([], torch.cat(target_parts))
    baseline = torch.cat(baseline_parts)
    solved = torch.cat(variant_parts["solved"])
    outcomes = {name: measure(labels, torch.cat(parts)) for name, parts in variant_parts.items()}
    random_outcomes = [row["accuracy"] for name, row in outcomes.items() if name.startswith("random_")]
    report = {
        "model_revision": model_path.name,
        "dataset": "google/boolq",
        "fit_questions": len(fit_rows),
        "calibration_questions": len(calibration_rows),
        "development_questions": len(test_rows),
        "exact_train_passage_overlap_excluded": exact_overlap,
        "calibration_trials": trials,
        "selected_scale": selected,
        "nonzero_probe_scale": probe,
        "base": measure(labels, baseline),
        "selected": statistics(labels, baseline, solved if selected else baseline),
        "nonzero_probe": statistics(labels, baseline, solved),
        "nonzero_probe_variants": outcomes,
        "random_mean_accuracy": sum(random_outcomes) / len(random_outcomes),
        "random_max_accuracy": max(random_outcomes),
        "random_at_least_as_good": sum(score >= outcomes["solved"]["accuracy"] for score in random_outcomes),
        "live_forward_audits": audits,
        "nonzero_probe_preservation": check_preservation(experiment, key, probe),
    }
    if experiment.weight_digest() != base_digest:
        raise RuntimeError("The original matrix changed")
    report["base_matrix_unchanged"] = True
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"BoolQ results saved to {path}", flush=True)


def run_native_probe(model_path, data, device):
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["model_revision"] != model_path.name:
        raise ValueError("Use the same checkpoint that produced the main BoolQ report")
    scale = report["nonzero_probe_scale"]
    fit, _, test, _ = load_data(data)
    experiment = Experiment(model_path, device)
    key = experiment.solve(*fitting_inputs(experiment, fit))

    tokenizer = experiment.teacher.tokenizer
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in (" No", " Yes")]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("Direct Yes/No scoring requires single-token labels")
    output_weights = experiment.model.get_output_embeddings().weight.detach()
    experiment.answer_rows = output_weights[[ids[0] for ids in token_ids]]
    sample = sorted(random.Random(911).sample(test, 512), key=lambda row: len(row["passage"]))
    before_parts, patched_parts = [], []
    for start in range(0, len(sample), 128):
        rows = sample[start:start + 128]
        prompts = [
            "Answer the question using the passage. Answer with Yes or No.\n\n"
            f"Passage: {row['passage']}\n\nQuestion: {row['question']}?"
            for row in rows
        ]
        cases = Cases(prompts, torch.tensor([row["answer"] for row in rows]))
        base, capture = experiment.forward(cases, capture=True, answers_only=True, batch_size=batch_size(rows))
        patched = experiment.score_cached_answers(capture, key, scale)
        before_parts.append((base[:, 1] > base[:, 0]) == cases.expected_b)
        patched_parts.append((patched[:, 1] > patched[:, 0]) == cases.expected_b)
        print(f"Direct Yes/No: {start + len(rows)}/{len(sample)} questions", flush=True)
    before = torch.cat(before_parts)
    patched = torch.cat(patched_parts)
    result = {
        "questions": len(sample), "sample_seed": 911, "scale": scale,
        "base_accuracy": float(before.float().mean()),
        "patched_accuracy": float(patched.float().mean()),
        "corrected_questions": int((patched & ~before).sum()),
        "regressed_questions": int((before & ~patched).sum()),
    }
    report["direct_yes_no_probe"] = result
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Direct Yes/No:", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted local LFM3-1.7B snapshot")
    parser.add_argument("--data", type=Path, default=OUTPUT / "data")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--random-count", type=int, default=RANDOM_COUNT)
    parser.add_argument("--native-probe", action="store_true", help="Check the fitted edit with direct Yes/No scoring")
    args = parser.parse_args()
    if args.random_count < 1:
        parser.error("--random-count must be positive")
    torch.set_num_threads(1)
    if args.native_probe:
        run_native_probe(args.model, args.data, args.device)
    else:
        run(args.model, args.data, args.device, args.random_count)


if __name__ == "__main__":
    main()
