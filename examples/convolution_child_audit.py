"""Cross adjective and sentence-frame shifts on a frozen standalone child."""
import argparse
import json
from pathlib import Path

from examples.convolution_child import SPLITS, load_file, score, Tokenizer
from examples.family import Teacher, Task


def run(model, folder):
    teacher = Teacher(model, "cuda")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    child = load_file(folder / "child.safetensors")
    task = Task("How was the service actually received, regardless of what the customer expected?",
                ("negative", "positive"), (), (), ())
    results = {}
    for words, (pos, neg, _) in SPLITS.items():
        for frames, (_, _, templates) in SPLITS.items():
            messages, labels = [], []
            for i, template in enumerate(templates):
                for p in pos:
                    for n in neg:
                        messages.extend([template.format(p,n), template.format(n,p)])
                        labels.extend([bool(i), not bool(i)])
            parent_scores = teacher.predict(messages, task)
            child_scores = [score(child,tokenizer.encode(s,add_special_tokens=False).ids) for s in messages]
            record = {"messages":len(messages)}
            for name, scores in (("parent_fixed_AB",parent_scores),("child",child_scores)):
                correct = [bool(v>0)==y for v,y in zip(scores,labels)]
                record[name] = {"correct":sum(correct),"both_correct":sum(a and b for a,b in zip(correct[::2],correct[1::2]))}
            results[f"{words}_adjectives__{frames}_frames"] = record
    print(json.dumps(results,indent=2))
    (folder / "crossed.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("--folder",type=Path,default=Path("artifacts/convolution_child"))
    args = parser.parse_args()
    run(args.model,args.folder)
