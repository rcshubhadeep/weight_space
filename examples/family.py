"""Distill and route small text classifiers using a local language model."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import median
from time import perf_counter

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


OUTPUT = Path("artifacts/family")
FEATURES = 512
GENERATIONS = ("child", "grandchild")


@dataclass(frozen=True)
class Task:
    question: str
    labels: tuple[str, str]
    phrases: tuple[tuple[str, ...], tuple[str, ...]]
    novel_phrases: tuple[tuple[str, ...], tuple[str, ...]]
    calibration_phrases: tuple[tuple[str, ...], tuple[str, ...]]


TASKS = {
    "sentiment": Task(
        "What is the customer's sentiment toward the service?",
        ("negative", "positive"),
        (("the support was awful", "I hated the experience", "the service was disappointing",
          "I am unhappy with the service", "the service was a disaster", "the service was terrible"),
         ("the support was excellent", "I loved the experience", "the service was fantastic",
          "I am delighted with the service", "the service was wonderful", "the service was outstanding")),
        (("I felt let down", "I regret using this company", "I would give this zero stars",
          "I would warn my friends away"),
         ("I am over the moon", "I would recommend this company", "I would give this five stars",
          "I look forward to coming back")),
        (("this left a bad impression", "the experience did not meet expectations"),
         ("this left a great impression", "the experience exceeded expectations")),
    ),
    "urgency": Task(
        "How soon does the customer need a response or action?",
        ("can wait", "needs immediate attention"),
        (("this can wait until next week", "there is no rush", "reply whenever convenient",
          "handle this at your leisure", "it is fine to wait several days", "a reply next month is fine"),
         ("please act immediately", "this is urgent", "please respond right now",
          "I need this fixed within the hour", "do not delay the response", "I need help at once")),
        (("feel free to put it on the backlog", "there is no deadline on my end",
          "leave it for another day", "I have plenty of time"),
         ("drop everything and look at this", "every minute counts", "I cannot wait any longer",
          "I need an answer before this call ends")),
        (("please put this on hold", "this is low priority"),
         ("we need a rapid intervention", "treat this as top priority")),
    ),
}

TEMPLATES = (
    ("Ticket {id}: {a}. {b}.", "Order {id}. The customer says: {a}. Also, {b}.",
     "Message {id}: {a}. As for timing, {b}."),
    ("Following up on case {id}: {b}. My feedback is that {a}.",
     "Regarding request {id}, one thing first: {b}. Separately, {a}."),
    ("Customer note for {id} — '{a}.' They added, '{b}.'",
     "Case {id} update. Timing: {b}. Feedback: {a}."),
)


def make_examples(count, seed, templates, phrase_sets=("phrases", "phrases")):
    rng = random.Random(seed)
    labels = [(i % 2, i // 2 % 2) for i in range(count)]
    rng.shuffle(labels)
    messages = []
    for a, b in labels:
        phrases = [
            rng.choice(getattr(task, phrase_set)[label])
            for task, label, phrase_set in zip(TASKS.values(), (a, b), phrase_sets)
        ]
        messages.append(rng.choice(templates).format(id=rng.randrange(10000, 99999), a=phrases[0], b=phrases[1]))
    return messages, torch.tensor(labels)


def encode(messages):
    features = torch.zeros(len(messages), FEATURES)
    for row, message in zip(features, messages):
        words = re.findall(r"[a-z]+", message.lower())
        grams = words + [" ".join(pair) for pair in zip(words, words[1:])]
        for gram in grams:
            slot = int.from_bytes(hashlib.blake2s(gram.encode(), digest_size=4).digest(), "little") % FEATURES
            row[slot] = 1
    return features


def specialist(width):
    return nn.Sequential(nn.Linear(FEATURES, width), nn.ReLU(), nn.Linear(width, 1))


def train(features, targets, width, seed):
    torch.manual_seed(seed)
    model = specialist(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(180):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(features).squeeze(-1), targets)
        loss.backward()
        optimizer.step()
    return model.eval()


class Teacher:
    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=True,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).eval().to(device)
        self.answers = [self.tokenizer.encode(label, add_special_tokens=False) for label in (" A", " B")]
        if any(len(ids) != 1 for ids in self.answers):
            raise ValueError("This tokenizer must encode each answer as a single token")
        self.answers = [ids[0] for ids in self.answers]

    @torch.inference_mode()
    def predict(self, messages, task, batch_size=12):
        result = []
        for start in range(0, len(messages), batch_size):
            prompts = []
            for message in messages[start:start + batch_size]:
                instruction = (
                    f"{task.question} A = {task.labels[0]}. B = {task.labels[1]}. "
                    f"Choose only A or B.\n\n{message}"
                )
                prompts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ) + "Answer:")
            batch = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            logits = self.model(**batch.to(self.device), use_cache=False, logits_to_keep=1).logits[:, -1]
            answer_logits = logits[:, self.answers].float()
            if not answer_logits.isfinite().all():
                raise RuntimeError("Teacher returned a non-finite answer logit")
            result.append((answer_logits[:, 1] - answer_logits[:, 0]).cpu())
        return torch.cat(result)


def measure(logits, gold, teacher):
    predictions = logits >= 0
    return {
        "gold_accuracy": (predictions == gold).float().mean().item(),
        "teacher_agreement": (predictions == (teacher >= 0)).float().mean().item(),
        "n": gold.numel(),
    }


def vocabulary_coverage(features, known):
    return features[:, known].sum(dim=1) / features.sum(dim=1).clamp_min(1)


def calibrate_gate(model, features, gold, known):
    """Select the broadest validation subset with at most 5% observed error."""
    coverage = vocabulary_coverage(features, known)
    correct = (model(features).squeeze(-1) >= 0) == gold
    for threshold in coverage.unique(sorted=True):
        accepted = coverage >= threshold
        if accepted.sum() >= 16 and correct[accepted].float().mean() >= 0.95:
            return float(threshold)
    return None


def routing_metrics(scores, teacher_scores, labels, coverage, threshold):
    accepted = coverage >= threshold if threshold is not None else torch.zeros_like(coverage, dtype=torch.bool)
    routed = torch.where(accepted, scores, teacher_scores)
    return {
        "specialist_count": int(accepted.sum()),
        "specialist_accuracy": float(((scores[accepted] >= 0) == labels[accepted]).float().mean())
        if accepted.any() else None,
        "accuracy_with_teacher_fallback": float(((routed >= 0) == labels).float().mean()),
    }


def automatic_metrics(scores, teacher_scores, labels, coverage, thresholds):
    routed = teacher_scores.clone()
    unassigned = torch.ones_like(labels, dtype=torch.bool)
    counts = {}
    for name in GENERATIONS:
        threshold = thresholds[name]
        accepted = unassigned & (coverage >= threshold) if threshold is not None else torch.zeros_like(unassigned)
        routed[accepted] = scores[name][accepted]
        unassigned &= ~accepted
        counts[name] = int(accepted.sum())
    counts["teacher"] = int(unassigned.sum())
    return {"counts": counts, "accuracy": float(((routed >= 0) == labels).float().mean())}


def median_ms(predict, messages):
    with torch.inference_mode():
        predict(messages[0])
        durations = []
        for message in messages:
            start = perf_counter()
            predict(message)
            durations.append((perf_counter() - start) * 1000)
    return median(durations)


def run(path, device):
    torch.set_num_threads(1)
    teacher = Teacher(path, device)
    train_text, train_labels = make_examples(128, 412, TEMPLATES[0])
    apprentice_text, _ = make_examples(128, 413, TEMPLATES[1])
    familiar_cal, familiar_labels = make_examples(64, 416, TEMPLATES[1])
    novel_cal, novel_labels = make_examples(
        64, 417, TEMPLATES[1], ("calibration_phrases", "calibration_phrases"),
    )
    cal_features = encode(familiar_cal + novel_cal)
    cal_labels = torch.cat((familiar_labels, novel_labels))
    eval_sets = {
        "new_templates": make_examples(96, 414, TEMPLATES[2]),
        "new_vocabulary": make_examples(64, 415, TEMPLATES[2], ("novel_phrases", "novel_phrases")),
    }
    training_features = encode(train_text)
    known = training_features.bool().any(dim=0)
    apprentice_features = encode(apprentice_text)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "teacher": {
            "checkpoint": str(path),
            "parameters": sum(p.numel() for p in teacher.model.parameters()),
            "device": device,
            "dtype": str(teacher.model.dtype),
        },
        "tasks": {},
    }
    for column, (key, task) in enumerate(TASKS.items()):
        print(f"Teaching {key}", flush=True)
        soft = teacher.predict(train_text, task).sigmoid()
        child = train(training_features, soft, width=16, seed=100 + column)
        with torch.no_grad():
            apprentice_targets = child(apprentice_features).squeeze(-1).sigmoid()
        grandchild = train(apprentice_features, apprentice_targets, width=8, seed=200 + column)
        # This comparison uses labels that the two descendants never train on.
        supervised = train(training_features, train_labels[:, column].float(), width=16, seed=100 + column)
        models = {"child": child, "grandchild": grandchild, "supervised_control": supervised}
        with torch.no_grad():
            thresholds = {
                name: calibrate_gate(models[name], cal_features, cal_labels[:, column], known)
                for name in GENERATIONS
            }
        metrics = {}
        mixed_phrases = tuple("novel_phrases" if index == column else "phrases" for index in range(len(TASKS)))
        mixed = make_examples(64, 418, TEMPLATES[2], mixed_phrases)
        for split, (messages, labels) in {**eval_sets, "novel_target_only": mixed}.items():
            scores = teacher.predict(messages, task)
            gold = labels[:, column]
            features = encode(messages)
            metrics[split] = {"teacher": measure(scores, gold, scores)}
            with torch.no_grad():
                coverage = vocabulary_coverage(features, known)
                predictions = {name: model(features).squeeze(-1) for name, model in models.items()}
                for name, predicted in predictions.items():
                    metrics[split][name] = measure(predicted, gold, scores)
                    if name in thresholds:
                        metrics[split][name]["routing"] = routing_metrics(
                            predicted, scores, gold, coverage, thresholds[name],
                        )
                metrics[split]["automatic_routing"] = automatic_metrics(
                    predictions, scores, gold, coverage, thresholds,
                )
        lineage = {}
        for name, parent, width in (("child", "teacher", 16), ("grandchild", "child", 8)):
            filename = f"{key}_{name}.safetensors"
            model = models[name]
            save_file(model.state_dict(), str(OUTPUT / filename))
            lineage[name] = {
                "parent": parent,
                "width": width,
                "parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint": filename,
                "min_vocabulary_coverage": thresholds[name],
            }
        timing_text = eval_sets["new_templates"][0][:12]
        report["tasks"][key] = {
            "lineage": lineage, "held_out": metrics,
            "batch_one_median_ms": {
                "teacher": median_ms(lambda s: teacher.predict([s], task), timing_text),
                "child": median_ms(lambda s: child(encode([s])), timing_text),
                "grandchild": median_ms(lambda s: grandchild(encode([s])), timing_text),
            },
        }
        print(json.dumps(metrics), flush=True)
    report["known_feature_indices"] = known.nonzero().flatten().tolist()
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Family lineage and measurements to {OUTPUT}", flush=True)


def route(task, message, generation, teacher_path, device):
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    lineage = report["tasks"][task]["lineage"]
    features = encode([message])
    known = torch.zeros(FEATURES, dtype=torch.bool)
    known[report["known_feature_indices"]] = True
    coverage = float(vocabulary_coverage(features, known).item())
    result = {"task": task, "generation": generation, "vocabulary_coverage": coverage}
    candidates = GENERATIONS if generation == "auto" else (generation,)
    for name in candidates:
        record = lineage[name]
        threshold = record["min_vocabulary_coverage"]
        if threshold is None or coverage < threshold:
            continue
        model = specialist(record["width"])
        model.load_state_dict(load_file(str(OUTPUT / record["checkpoint"])))
        with torch.inference_mode():
            positive = float(model(features).sigmoid().item())
        result.update(
            route="specialist", generation=name,
            prediction=TASKS[task].labels[int(positive >= 0.5)], positive_score=positive,
        )
        break
    else:
        result["route"] = "defer"
        if teacher_path is not None:
            logits = Teacher(teacher_path, device).predict([message], TASKS[task])
            result.update(route="teacher", generation="teacher", prediction=TASKS[task].labels[int(logits.item() >= 0)])
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--teacher", type=Path, required=True, help="Trusted local checkpoint directory")
    experiment.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    router = commands.add_parser("route")
    router.add_argument("task", choices=TASKS)
    router.add_argument("message")
    router.add_argument("--generation", default="auto", choices=("auto", *GENERATIONS))
    router.add_argument("--teacher", type=Path, help="Trusted local fallback checkpoint")
    router.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.command == "run":
        run(args.teacher, args.device)
    else:
        route(args.task, args.message, args.generation, args.teacher, args.device)


if __name__ == "__main__":
    main()
