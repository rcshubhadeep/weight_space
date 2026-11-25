"""Compile two local classifiers from LFM input embeddings. Run as a module."""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer


TASK_NAMES = ("sentiment", "urgency")
OUTPUT = Path("artifacts/embedding_children")


def ids(tokenizer, text):
    return tokenizer.encode(" " + text.lower().strip(), add_special_tokens=False).ids


def predict(folder, message):
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokens = ids(tokenizer, message)
    if not tokens:
        raise ValueError("Enter a nonempty message")
    meta = json.loads((folder / "children.json").read_text())
    for name in TASK_NAMES:
        table = (folder / (name + ".bin")).read_bytes()
        score = sum(table[i] - 128 for i in tokens) / len(tokens)
        child = meta[name]
        print(name, child["labels"][int(score > child["threshold"])],
              f"(signed margin {score - child['threshold']:+.2f})")


def compile_children(model, folder):
    import torch
    from torch.nn.functional import normalize
    from safetensors import safe_open
    from examples.family import TASKS, TEMPLATES

    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model / "tokenizer.json", folder / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    with safe_open(model / "model.safetensors", framework="pt", device="cpu") as source:
        emb = source.get_tensor("model.embed_tokens.weight").to(device="cuda", dtype=torch.float32)
    emb -= emb.mean(0)
    emb = normalize(emb)
    vocab = tokenizer.get_vocab_size()

    def cases(split, templates):
        sets = [getattr(TASKS[name], split) for name in TASK_NAMES]
        for s, u in product(range(2), repeat=2):
            for a, b, template in product(sets[0][s], sets[1][u], templates):
                yield ids(tokenizer, template.format(a=a, b=b, id=12345)), (s, u)

    cal = list(cases("calibration_phrases", TEMPLATES[1]))
    test = list(cases("novel_phrases", TEMPLATES[2]))
    meta, report = {}, {"model_revision": model.name, "calibration_messages": len(cal),
                        "evaluation_messages": len(test), "tasks": {}}
    for col, name in enumerate(TASK_NAMES):
        task = TASKS[name]
        weights = torch.zeros(vocab, device="cuda")
        for label, phrases in enumerate(task.phrases):
            for phrase in phrases:
                tokens = ids(tokenizer, phrase)
                indices = torch.tensor(tokens, device="cuda")
                weights.index_add_(0, indices, torch.full_like(indices, (2*label-1) /
                                   (len(phrases)*len(tokens)), dtype=weights.dtype))
        semantic = emb[:vocab] @ (weights @ emb[:vocab])
        results = {}
        for mode, scores in (("inherited", semantic), ("overlap", weights)):
            q = (127 * scores / scores.abs().max()).round().to(torch.int16).cpu().tolist()

            def score(row):
                tokens, _ = row
                return sum(q[i] for i in tokens) / len(tokens)

            observed = sorted(set(map(score, cal)))
            candidates = [observed[0]-1, *[(a+b)/2 for a,b in zip(observed, observed[1:])], observed[-1]]
            threshold = max(candidates, key=lambda t: (sum((score(row)>t)==row[1][col] for row in cal), -abs(t)))
            results[mode] = {"correct": sum((score(row)>threshold)==row[1][col] for row in test),
                             "threshold": threshold}
            if mode == "inherited":
                (folder / (name + ".bin")).write_bytes(bytes(x+128 for x in q))
                meta[name] = {"labels": task.labels, "threshold": threshold}
        report["tasks"][name] = results
    report["bytes_per_table"] = vocab
    report["shared_tokenizer_bytes"] = (folder / "tokenizer.json").stat().st_size
    for filename, data in (("children.json", meta), ("report.json", report)):
        (folder / filename).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compile", type=Path, metavar="LFM_SNAPSHOT")
    group.add_argument("--message")
    parser.add_argument("--folder", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.compile:
        compile_children(args.compile, args.folder)
    else:
        predict(args.folder, args.message)
