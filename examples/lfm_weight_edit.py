"""Write and verify a reversible rank-one patch to an LFM3 checkpoint in memory."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from examples.family import TASKS, TEMPLATES, Teacher, make_examples
from weight_space.low_rank import add_rank_one_branch, fit_key


PARAMETER = "model.layers.23.feed_forward.w2.weight"
OUTPUT = Path("artifacts/lfm_weight_edit")


@dataclass
class Cases:
    prompts: list[str]
    expected_b: torch.Tensor | None


def classification(split, task, phrase_sets=("phrases", "phrases"), count=32):
    messages, labels = make_examples(count, 7281 + split, TEMPLATES[split], phrase_sets)
    column = list(TASKS).index(task)
    spec = TASKS[task]
    prompts, gold = [], []
    for text, label in zip(messages, labels[:, column].tolist()):
        for reverse in (False, True):
            a, b = spec.labels[::-1] if reverse else spec.labels
            prompts.append(f"{spec.question} A = {a}. B = {b}. Choose only A or B.\n\n{text}")
            gold.append(label ^ reverse)
    return Cases(prompts, torch.tensor(gold))


def general_cases(held_out):
    topics = (
        ("the Roman Empire", "a violin", "the Moon", "a paper airplane", "an oak tree", "gravity")
        if held_out else
        ("a bicycle", "a sunflower", "a chessboard", "the ocean", "a calculator", "a violin bow")
    )
    prompts = [f"Explain how {topic} works or what it is in one sentence." for topic in topics]
    prompts.extend(
        ["Complete this sentence: The capital of France is", "What is 13 plus 29?",
         "Translate 'Good morning' into French.", "Write the start of a polite resignation letter."]
        if held_out else
        ["Complete this sentence: The capital of Japan is", "What is 7 plus 18?",
         "Translate 'Thank you' into German.", "Write the start of a formal invitation."]
    )
    return Cases(prompts, None)


class Experiment:
    def __init__(self, model_path, device):
        self.teacher = Teacher(model_path, device)
        self.model = self.teacher.model
        block = self.model.model.layers[23]
        self.weight = self.model.get_parameter(PARAMETER)
        self.consumers = (block.feed_forward.w2, block.ffn_norm, block.post_ffn_norm)
        self.post_norm = block.post_ffn_norm
        self.final_norm = self.model.model.embedding_norm
        self.answer_rows = self.model.get_output_embeddings().weight.detach()[self.teacher.answers]
        self.value = F.normalize((self.answer_rows[1] - self.answer_rows[0]).float(), dim=0)

    @torch.inference_mode()
    def forward(self, cases, capture=False, answers_only=False, batch_size=16):
        buffers = [[] for _ in self.consumers]

        def record(buffer):
            def hook(_module, args):
                buffer.append(args[0][:, -1].float())
            return hook

        handles = [module.register_forward_pre_hook(record(buffer))
                   for module, buffer in zip(self.consumers, buffers)] if capture else []
        logits = []
        try:
            for start in range(0, len(cases.prompts), batch_size):
                prompts = []
                for text in cases.prompts[start:start + batch_size]:
                    prompt = self.teacher.tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    prompts.append(prompt + ("Answer:" if cases.expected_b is not None else ""))
                batch = self.teacher.tokenizer(
                    prompts, padding=True, return_tensors="pt", add_special_tokens=False,
                ).to(self.teacher.device)
                if answers_only:
                    hidden = self.model.model(**batch, use_cache=False).last_hidden_state[:, -1]
                    batch_logits = F.linear(hidden, self.answer_rows)
                else:
                    batch_logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1]
                logits.append(batch_logits.float().cpu())
        finally:
            for handle in handles:
                handle.remove()
        result = torch.cat(logits)
        if not result.isfinite().all():
            raise RuntimeError("Model returned non-finite logits")
        return result, tuple(torch.cat(buffer) for buffer in buffers) if capture else ()

    def margins(self, logits):
        a, b = (0, 1) if logits.shape[-1] == 2 else self.teacher.answers
        return logits[:, b] - logits[:, a]

    @torch.inference_mode()
    def score_cached_answers(self, capture, key, scale):
        keys, residual, raw_ffn = capture
        dtype = self.weight.dtype
        patched_ffn = add_rank_one_branch(keys, raw_ffn, self.value, key, scale)
        ffn = self.post_norm(patched_ffn.to(dtype))
        hidden = self.final_norm(residual.to(dtype) + ffn)
        return F.linear(hidden, self.answer_rows).float().cpu()

    @contextmanager
    def attached(self, key, scale):
        def hook(_module, args, output):
            return add_rank_one_branch(args[0], output, self.value, key, scale).to(output.dtype)

        handle = self.consumers[0].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def weight_digest(self):
        raw = self.weight.detach().cpu().contiguous().view(torch.uint8)
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    @torch.inference_mode()
    def solve(self, target, target_logits, capture, protect=None):
        keys, residual, raw_ffn = capture
        dtype = self.weight.dtype

        def suffix(alpha):
            ffn = self.post_norm((raw_ffn + alpha * self.value).to(dtype))
            final = self.final_norm(residual.to(dtype) + ffn)
            return final.float() @ (self.answer_rows[1] - self.answer_rows[0]).float()

        # Estimate sensitivity through the real normalization and readout, using two forwards.
        slope = (suffix(0.25) - suffix(-0.25)) / 0.5
        signed = (2 * target.expected_b - 1).to(keys.device)
        before = self.margins(target_logits).to(keys.device)
        wanted = signed * (2 - signed * before).clamp_min(0)
        shift = (wanted * slope / (slope.square() + 0.05)).clamp(-15, 15)
        return fit_key(keys, shift, protect)

    @contextmanager
    def written(self, key, scale, value=None):
        with torch.no_grad():
            original = self.weight.detach().clone()
            value = self.value if value is None else value
            updated = (original.float() + scale * torch.outer(value, key)).to(self.weight.dtype)
            changed = int(updated.ne(original).sum())
            try:
                self.weight.copy_(updated)
                yield changed
            finally:
                self.weight.copy_(original)
                if not torch.equal(self.weight, original):
                    raise RuntimeError("Weight restoration failed")

    def accuracy(self, cases, logits):
        return float(((self.margins(logits) > 0) == cases.expected_b).float().mean())


def retention(before, after):
    log_before = before.log_softmax(dim=-1)
    log_after = after.log_softmax(dim=-1)
    kl = (log_before.exp() * (log_before - log_after)).sum(dim=-1).clamp_min(0)
    return {
        "top1_agreement": float((before.argmax(dim=-1) == after.argmax(dim=-1)).float().mean()),
        "mean_next_token_kl_nats": float(kl.mean()),
        "max_next_token_kl_nats": float(kl.max()),
    }


def test_suites():
    return {
        "urgency_new_templates": classification(2, "urgency"),
        "urgency_new_phrases": classification(2, "urgency", ("novel_phrases", "novel_phrases")),
        "sentiment": classification(2, "sentiment", ("novel_phrases", "novel_phrases")),
        "general_next_token": general_cases(True),
    }


def evaluate_edit(experiment, tests, baseline, key, scale, value=None):
    outcomes = {}
    with experiment.written(key, scale, value) as changed:
        for name, cases in tests.items():
            after, _ = experiment.forward(cases)
            outcomes[name] = {"n": len(cases.prompts), **retention(baseline[name], after)}
            if cases.expected_b is not None:
                outcomes[name]["base_accuracy"] = experiment.accuracy(cases, baseline[name])
                outcomes[name]["edited_accuracy"] = experiment.accuracy(cases, after)
    return changed, outcomes


def run(path, device):
    experiment = Experiment(path, device)
    with torch.inference_mode():
        target = classification(0, "urgency")
        protected = classification(0, "sentiment", count=16)
        training_logits, captured = experiment.forward(target, capture=True)
        _, protected_capture = experiment.forward(protected, capture=True)
        _, general_capture = experiment.forward(general_cases(False), capture=True)
        protect = torch.cat((protected_capture[0], general_capture[0]))
        guarded = experiment.solve(target, training_logits, captured, protect)
        unguarded = experiment.solve(target, training_logits, captured)
        unguarded = F.normalize(unguarded, dim=0) * guarded.norm()

    validation = classification(1, "urgency", ("phrases", "calibration_phrases"))
    validation_guard = classification(1, "sentiment", count=16)
    valid_base, _ = experiment.forward(validation)
    guard_base, _ = experiment.forward(validation_guard)
    trials = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        with experiment.written(guarded, scale):
            trial_logits, _ = experiment.forward(validation)
            trial_guard, _ = experiment.forward(validation_guard)
        trials.append({
            "scale": scale,
            "urgency_accuracy": experiment.accuracy(validation, trial_logits),
            "sentiment_accuracy": experiment.accuracy(validation_guard, trial_guard),
            **retention(guard_base, trial_guard),
        })
    feasible = [row for row in trials if row["sentiment_accuracy"] >= trials[0]["sentiment_accuracy"] - 0.03
                and row["mean_next_token_kl_nats"] <= 0.05]
    selected = max(feasible, key=lambda row: (row["urgency_accuracy"], -row["scale"]))
    scale = selected["scale"]
    print(f"Validation selected scale {scale}: {trials[0]['urgency_accuracy']:.1%} -> "
          f"{selected['urgency_accuracy']:.1%}", flush=True)

    tests = test_suites()
    baseline = {name: experiment.forward(cases)[0] for name, cases in tests.items()}
    report = {
        "checkpoint": str(path),
        "parameter": PARAMETER,
        "parameter_shape": list(experiment.weight.shape),
        "dtype": str(experiment.weight.dtype),
        "selected_scale": scale,
        "validation_trials": trials,
        "held_out": {},
    }
    generator = torch.Generator().manual_seed(726)
    random_keys = [
        F.normalize(torch.randn(guarded.numel(), generator=generator), dim=0).to(device) * guarded.norm()
        for _ in range(3)
    ]
    variants = {"guarded": guarded, "unguarded_norm_matched": unguarded}
    variants.update({f"random_key_{i}": key for i, key in enumerate(random_keys, 1)})
    for variant, key in variants.items():
        changed, outcomes = evaluate_edit(experiment, tests, baseline, key, scale)
        report["held_out"][variant] = {
            "changed_entries_after_rounding": changed,
            "delta_frobenius_before_rounding": float(scale * key.norm()),
            "outcomes": outcomes,
        }
        print(variant, json.dumps(outcomes), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    patch = OUTPUT / "patch.safetensors"
    save_file({"value": experiment.value.cpu(), "key": guarded.cpu()}, str(patch), metadata={
        "parameter": PARAMETER, "checkpoint": path.name, "operation": "add scale * outer(value, key) in float32",
        "scale": str(scale), "base_sha256": experiment.weight_digest(),
    })
    report["patch_bytes"] = patch.stat().st_size
    report["restored"] = True
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def replay(path, device, patch):
    with safe_open(str(patch), framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
        if metadata.get("checkpoint") != path.name or metadata.get("parameter") != PARAMETER:
            raise ValueError("This patch targets a different checkpoint or parameter")
        key = saved.get_tensor("key").to(device=device, dtype=torch.float32)
        value = saved.get_tensor("value").to(device=device, dtype=torch.float32)
        scale = float(metadata["scale"])

    experiment = Experiment(path, device)
    if (value.ndim, key.ndim) != (1, 1) or (value.numel(), key.numel()) != tuple(experiment.weight.shape):
        raise ValueError("The patch factors do not match the target matrix")
    if metadata.get("base_sha256") != experiment.weight_digest():
        raise ValueError("The base weight matrix differs from the patch's original matrix")
    if not math.isfinite(scale) or not value.isfinite().all() or not key.isfinite().all():
        raise ValueError("Patch factors and scale must be finite")
    tests = test_suites()
    baseline = {name: experiment.forward(cases)[0] for name, cases in tests.items()}
    changed, outcomes = evaluate_edit(experiment, tests, baseline, key, scale, value)
    print(json.dumps({
        "changed_entries_after_rounding": changed, "outcomes": outcomes, "restored": True,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted local LFM3-1.7B snapshot")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--patch", type=Path, help="Replay a saved weight patch against the held-out suites")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.patch:
        replay(args.model, args.device, args.patch)
    else:
        run(args.model, args.device)


if __name__ == "__main__":
    main()
