"""Check the hashes and internal consistency of a frozen evidence snapshot."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results" / "2026-11-25"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_digest(document):
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_replay(actual, original):
    expected = original["held_out"]["guarded"]
    for field in ("changed_entries_after_rounding", "outcomes"):
        require(actual[field] == expected[field], f"Replay mismatch: {field}")
    require(actual["restored"], "Replay did not restore the original matrix")


def check_results(reports):
    combined = reports["boolq_sst2_composition"]
    for task, n in (("boolq", 6540), ("sst2", 872)):
        validation = combined["validation"][task]
        require(validation["prompts"] == n, "Composition split size")
        require(len(validation["live_agreement"]) == 2, "Composition live audit coverage")
        for matching, total in validation["live_agreement"]:
            require(0 <= matching <= total and total > 0, "Composition live audit count")
        require(all(0 <= v <= n for v in validation["correct"].values()), "Composition correct count")
    child = reports["sst2_child"]
    edit = reports["sst2_liquid_edit"]
    require(child["input_sha256"] == edit["input_sha256"], "SST-2 input hashes")
    require(child["validation"] == edit["validation"] == 872, "SST-2 split")
    for scores in edit["systems"].values():
        for run in (scores, scores["full_head_replay"]):
            require(run["edited"] - run["base"] == run["fixed"] - run["broken"], "SST-2 paired edit counts")
    require(child["systems"]["parent_fixed_AB"]["correct"] == edit["systems"]["fixed_AB"]["full_head_replay"]["base"],
            "SST-2 full-head parent reproduction")
    boolq = reports["boolq"]
    controls = [row["accuracy"] for name, row in boolq["nonzero_probe_variants"].items()
                if name.startswith("random_")]
    solved = boolq["nonzero_probe_variants"]["solved"]["accuracy"]
    require(len(controls) == 32 and max(controls) == boolq["random_max_accuracy"], "BoolQ random control summary")
    require(sum(score >= solved for score in controls) == boolq["random_at_least_as_good"], "BoolQ random comparison")
    require(boolq["development_questions"] == boolq["selected"]["documents"] == 3270, "BoolQ held-out size")
    require(boolq["selected"]["patched_accuracy"] == solved, "BoolQ selected score")
    require(boolq["base_matrix_unchanged"], "BoolQ matrix integrity")
    audits = boolq["live_forward_audits"]
    require(sum(audit["prompts"] for audit in audits) == 70, "BoolQ audit size")
    require(all(audit["matching_choices"] == audit["prompts"] for audit in audits), "BoolQ live audit agreement")
    native = boolq["direct_yes_no_probe"]
    delta = native["patched_accuracy"] - native["base_accuracy"]
    require(math.isclose(delta * native["questions"], native["corrected_questions"] - native["regressed_questions"]),
            "BoolQ direct Yes/No paired counts")

    deletion = reports["deletion_composition"]
    both = [row for row in deletion["runs"] if row["passes"]["a"] and row["passes"]["b"]]
    failures = [row for row in both if not row["passes"]["union"]]
    require(len(both) == deletion["summary"]["both_individual_pass"], "Deletion individual pass count")
    require([row["seed"] for row in failures] == deletion["summary"]["disjoint_union_failure_seeds"],
            "Deletion failure seeds")
    for row in failures:
        a, b = (set(row["deletions"][name]) for name in ("a", "b"))
        joint = row["deletions"]["joint"]
        require(not a & b and a | b == set(row["deletions"]["union"]), "Disjoint-mask union")
        require(row["passes"]["joint"] and len(joint) == len(a | b), "Matched-budget joint recovery")

    extraction = reports["lfm_extraction_benchmark"]
    require(extraction["base_matrix_unchanged"], "Extraction matrix integrity")
    for split, outcomes in extraction["held_out"].items():
        constrained = outcomes["constrained"]
        require(constrained["documents"] == extraction["source_splits"][split]["eligible_documents"],
                f"Extraction split size for {split}")
        require(constrained["live_forward_audit"]["agreement"] == 1.0, f"Extraction live audit for {split}")
    require(reports["lfm_weight_edit"]["restored"], "Saved direct-write restoration result")
    if "lfm_weight_edit_replay" in reports:
        check_replay(reports["lfm_weight_edit_replay"], reports["lfm_weight_edit"])
    if "lfm_patch_composition" in reports:
        composition = reports["lfm_patch_composition"]
        require(composition["base_matrix_unchanged"], "Composition matrix integrity")
        for variant in composition["variants"].values():
            for row in variant["held_out"].values():
                audit = row["live_audit"]
                require(audit["prompts"] == audit["matching_choices"], "Composition live audit agreement")
                require(row["fp16_answer_changes_among_margin_check_passes"] == 0,
                        "Unexpected FP16 answer change among margin-check passes")
        probe = composition["variants"]["cross_protected_nonzero_probe"]
        require(all(row["scale"] > 0 for row in probe["calibration"].values()), "Nonzero composition probe")
        require(all(row["answer_changes_from_own"] == 0 for row in probe["fit_anchors"].values()),
                "Cross-protected anchor answers")
    if "lfm_algebraic_boundaries" in reports:
        algebraic = reports["lfm_algebraic_boundaries"]
        require(algebraic["base_matrix_unchanged"], "Algebraic run matrix integrity")
        for name, variant in algebraic["variants"].items():
            original = reports["lfm_patch_composition"]["variants"][name]
            require(variant["fit_anchors"] == original["fit_anchors"], "Replayed composition anchors")
            require(variant["calibration"] == original["calibration"], "Replayed composition calibration")
            for task, row in variant["held_out"].items():
                original_row = original["held_out"][task]
                require({k: v for k, v in row.items() if k != "algebraic_boundary"} == original_row,
                        "Replayed held-out composition results")
                boundary = row["algebraic_boundary"]
                stable = row["real_arithmetic_margin_check_passes"] + boundary["additional_to_margin_bound"]
                require(stable - boundary["margin_bound_only"] == boundary["stable_through_full_strength"],
                        "Algebraic protection count")
                require(boundary["checked_backoff_fp16_changes_from_own"] == 0, "Checked backoff choices")
                for witness in boundary["first_boundary_examples"]:
                    low, high = witness["first_tie_bracket"]
                    require(0 <= low <= high <= 1 and high - low <= 2**-18, "First-tie bracket")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--extraction-dir", type=Path)
    parser.add_argument("--boolq-dir", type=Path)
    parser.add_argument("--original-patch", type=Path, help="Check the original binary, if available")
    parser.add_argument("--replay", type=Path, help="Check metrics from a fresh direct-write patch replay")
    args = parser.parse_args()
    manifest = read_json(SNAPSHOT / "manifest.json")
    reports = {}
    for name, expected in manifest["reports_sha256"].items():
        document = read_json(SNAPSHOT / f"{name}.json")
        require(json_digest(document) == expected, f"Report hash mismatch: {name}")
        reports[name] = document
    check_results(reports)
    print(f"Verified {len(reports)} frozen report hashes and result consistency checks")

    for name in ("model", "extraction", "boolq"):
        directory = getattr(args, f"{name}_dir")
        if directory is None:
            continue
        files = manifest["inputs"][name]["files_sha256"]
        for filename, expected in files.items():
            path = directory / filename
            require(file_digest(path) == expected, f"Input hash mismatch: {path}")
        print(f"Verified {len(files)} {name} input file(s)")

    if args.original_patch:
        expected = manifest["original_direct_write_patch"]["sha256"]
        require(file_digest(args.original_patch) == expected, "Original patch hash mismatch")
        print("Verified the original direct-write patch hash")
    if args.replay:
        check_replay(read_json(args.replay), reports["lfm_weight_edit"])
        print("Replay matches all frozen direct-write metrics and reports exact restoration")


if __name__ == "__main__":
    main()
