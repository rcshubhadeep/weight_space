"""Compose two same-layer edits and test protection on anchors and unseen inputs."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from examples.lfm_extraction_benchmark import load_split, make_cases
from examples.lfm_weight_edit import Cases, Experiment, classification, general_cases
from weight_space.decision_boundary import build_decisions


OUTPUT = Path("artifacts/lfm_patch_composition")
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)


def observe(experiment, cases):
    return experiment.forward(cases, capture=True, answers_only=True)


def choose_scale(experiment, cases, capture, key, baseline):
    scores = [experiment.accuracy(cases, baseline)]
    for scale in SCALES[1:]:
        scores.append(experiment.accuracy(cases, experiment.score_cached_answers(capture, key, scale)))
    best = max(range(len(scores)), key=lambda i: (scores[i], -SCALES[i]))
    return SCALES[best], scores


def normalization_bound(x, radius, norm):
    dimension = x.shape[-1]
    lower = (x.norm(dim=-1) - radius).clamp_min(0)
    return dimension**0.5 / (lower.square() + dimension * norm.variance_epsilon).sqrt()


@torch.inference_mode()
def margin_envelope(experiment, capture, own_key, cross_key):
    """Bound the margin change around the own-patch state in real arithmetic."""
    keys, residual, raw = capture
    value = experiment.value
    own_raw = raw + (keys @ own_key).unsqueeze(-1) * value
    cross_radius = (keys @ cross_key).abs() * value.norm()
    post = experiment.post_norm
    final = experiment.final_norm
    post_weight = post.weight.float()
    post_rms = (own_raw.square().mean(dim=-1, keepdim=True) + post.variance_epsilon).rsqrt()
    final_input = residual + own_raw * post_rms * post_weight
    post_radius = cross_radius * normalization_bound(own_raw, cross_radius, post) * post_weight.abs().max()
    readout = (experiment.answer_rows[1] - experiment.answer_rows[0]).float() * final.weight.float()
    bound = post_radius * normalization_bound(final_input, post_radius, final) * readout.norm()
    final_rms = (final_input.square().mean(dim=-1) + final.variance_epsilon).rsqrt()
    margin = (final_input @ readout) * final_rms
    return cross_radius, margin, bound


@torch.inference_mode()
def audit_boundary(experiment, capture, own_key, cross_key, own_choice, combined_choice, certified):
    decisions = build_decisions(experiment, capture, own_key, cross_key)
    stable = torch.tensor([decision.stable_through(Fraction(1)) for decision in decisions])
    initial = torch.tensor([decision.sign_at(Fraction(0)) for decision in decisions])
    final = torch.tensor([decision.sign_at(Fraction(1)) for decision in decisions])
    boundaries = [None if safe else decision.first_boundary_bracket(1)
                  for safe, decision in zip(stable.tolist(), decisions, strict=True)]
    doses = [
        1.0 if interval is None else float(Fraction(63, 64) * interval[0])
        for interval in boundaries
    ]
    keys, residual, raw = capture
    own_alpha, cross_alpha = keys @ own_key, keys @ cross_key

    def rounded_choice(strength):
        alpha = own_alpha + strength.to(keys.device) * cross_alpha
        ffn = experiment.post_norm((raw + alpha.unsqueeze(-1) * experiment.value).to(experiment.weight.dtype))
        hidden = experiment.final_norm(residual.to(experiment.weight.dtype) + ffn)
        scores = F.linear(hidden, experiment.answer_rows).float().cpu()
        return torch.where(strength == 0, own_choice, experiment.margins(scores) > 0)

    checked_strength = torch.tensor(doses, dtype=keys.dtype)
    clipped_choice = rounded_choice(checked_strength)
    failures = clipped_choice != own_choice
    retry = failures
    for _ in range(8):
        if not retry.any():
            break
        checked_strength[retry] *= 0.5
        retry = rounded_choice(checked_strength) != own_choice
    checked_strength[retry] = 0
    checked_choice = rounded_choice(checked_strength)
    witness = [
        {"prompt_index": i, "first_tie_bracket": [float(t) for t in interval],
         "opposite_at_full_strength": bool(initial[i] * final[i] < 0)}
        for i, interval in enumerate(boundaries) if interval is not None
    ][:12]
    return {
        "stable_through_full_strength": int(stable.sum()),
        "additional_to_margin_bound": int((stable & ~certified).sum()),
        "margin_bound_only": int((~stable & certified).sum()),
        "ties_or_changes_with_same_strict_endpoint_signs": int((~stable & (initial * final > 0)).sum()),
        "fp16_endpoint_mismatches_at_zero": int(((initial > 0) != own_choice).sum()),
        "fp16_endpoint_mismatches_at_one": int(((final > 0) != combined_choice).sum()),
        "fp16_changes_with_exact_interval_protection": int((stable & (own_choice != combined_choice)).sum()),
        "clipping_mean_strength": sum(doses) / len(doses),
        "clipping_fp16_changes_from_own": int(failures.sum()),
        "checked_backoff_mean_strength": float(checked_strength.mean()),
        "checked_backoff_zero_strength": int((checked_strength == 0).sum()),
        "checked_backoff_fp16_changes_from_own": int((checked_choice != own_choice).sum()),
        "first_boundary_examples": witness,
    }


def evaluate(experiment, cases, observed, own_key, cross_key, batch_live=False, algebraic=False):
    base, capture = observed
    own = experiment.score_cached_answers(capture, own_key, 1.0)
    composed = experiment.score_cached_answers(capture, own_key + cross_key, 1.0)
    radius, margin, bound = margin_envelope(experiment, capture, own_key, cross_key)
    certified = (margin.abs() > bound).cpu()
    own_choice = experiment.margins(own) > 0
    combined_choice = experiment.margins(composed) > 0
    flip = own_choice != combined_choice
    result = {
        "prompts": len(cases.prompts),
        "base_accuracy": experiment.accuracy(cases, base),
        "own_patch_accuracy": experiment.accuracy(cases, own),
        "combined_accuracy": experiment.accuracy(cases, composed),
        "answer_changes_from_own": int(flip.sum()),
        "max_raw_cross_patch_norm": float(radius.max()),
        "median_raw_cross_patch_norm": float(radius.median()),
        "real_arithmetic_margin_check_passes": int(certified.sum()),
        "fp16_answer_changes_among_margin_check_passes": int((flip & certified).sum()),
    }
    if batch_live:
        audit = Cases(cases.prompts[:16], cases.expected_b[:16])
        with experiment.attached(own_key + cross_key, 1.0):
            live, _ = experiment.forward(audit, answers_only=True)
        result["live_audit"] = {
            "prompts": len(audit.prompts),
            "matching_choices": int(((experiment.margins(live) > 0) == combined_choice[:len(live)]).sum()),
        }
    if algebraic:
        result["algebraic_boundary"] = audit_boundary(
            experiment, capture, own_key, cross_key, own_choice, combined_choice, certified,
        )
    return result


def run(model, data, device, algebraic=False):
    _, train = load_split(data / "train.parquet", 1621)
    _, test = load_split(data / "test_unseen.parquet", 1622)
    fits = {"urgency": classification(0, "urgency"), "extraction": make_cases(train[:48])}
    validation = {
        "urgency": classification(1, "urgency", ("phrases", "calibration_phrases")),
        "extraction": make_cases(train[48:96]),
    }
    tests = {
        "urgency": classification(2, "urgency", ("novel_phrases", "novel_phrases"), count=64),
        "extraction": make_cases(test[:96]),
    }
    experiment = Experiment(model, device)
    base_digest = experiment.weight_digest()
    guard_cases = classification(0, "sentiment", count=16)
    guards = torch.cat((observe(experiment, guard_cases)[1][0],
                       experiment.forward(general_cases(False), capture=True)[1][0]))
    fit_observed = {name: observe(experiment, cases) for name, cases in fits.items()}
    valid_observed = {name: observe(experiment, cases) for name, cases in validation.items()}
    test_observed = {name: observe(experiment, cases) for name, cases in tests.items()}

    results = {}
    others = {"urgency": "extraction", "extraction": "urgency"}

    def record(name, keys, calibration):
        results[name] = {
            "calibration": calibration,
            "fit_anchors": {
                task: evaluate(experiment, cases, fit_observed[task], keys[task], keys[others[task]])
                for task, cases in fits.items()
            },
            "held_out": {
                task: evaluate(experiment, cases, test_observed[task], keys[task], keys[others[task]], True, algebraic)
                for task, cases in tests.items()
            },
        }
        print(name, json.dumps(results[name]), flush=True)

    for variant in ("independent", "cross_protected"):
        keys, calibration, probe_keys, probe_calibration = {}, {}, {}, {}
        for name, cases in fits.items():
            logits, capture = fit_observed[name]
            protect = guards
            if variant == "cross_protected":
                protect = torch.cat((guards, fit_observed[others[name]][1][0]))
            key = experiment.solve(cases, logits, capture, protect)
            base, valid_capture = valid_observed[name]
            scale, sweep = choose_scale(experiment, validation[name], valid_capture, key, base)
            keys[name] = key * scale
            calibration[name] = {
                "scale": scale, "accuracy_by_scale": dict(zip(SCALES, sweep)),
                "effective_key_norm": float(keys[name].norm()),
            }
            # Keep a nonzero diagnostic when calibration would leave one branch off.
            probe_scale = scale or max(SCALES[1:], key=lambda s: (sweep[SCALES.index(s)], -s))
            probe_keys[name] = key * probe_scale
            probe_calibration[name] = {
                **calibration[name], "scale": probe_scale, "effective_key_norm": float(probe_keys[name].norm()),
            }
        record(variant, keys, calibration)
        if variant == "cross_protected" and any(row["scale"] == 0 for row in calibration.values()):
            record("cross_protected_nonzero_probe", probe_keys, probe_calibration)

    if experiment.weight_digest() != base_digest:
        raise RuntimeError("The base matrix changed during the experiment")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"model_revision": model.name, "extraction_revision": data.name,
              "base_matrix_unchanged": True, "variants": results}
    filename = "algebraic_report.json" if algebraic else "report.json"
    (OUTPUT / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--algebraic", action="store_true", help="Audit exact decision thresholds on held-out prompts")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.model, args.data, args.device, args.algebraic)


if __name__ == "__main__":
    main()
