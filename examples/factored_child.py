"""Factor a local decision into an outcome cue and a sentiment direction."""
import argparse
import json
from pathlib import Path
import re

import torch
from torch.nn.functional import normalize
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tokenizers import Tokenizer

from examples.convolution_child import rows, SPLITS


def describe(tok, text):
    return [tok.encode(" "+word, add_special_tokens=False).ids for word in re.findall(r"[a-z]+",text.lower())]


def decide(table, words):
    if len(words)<2:
        return 0.
    values = torch.stack([table[w].mean(0) for w in words])
    return float((values[:-1,0] * values[1:,1]).sum())


def run(model, folder):
    tok = Tokenizer.from_file(str(folder/'tokenizer.json'))
    with safe_open(model/'model.safetensors', framework='pt') as source:
        e = source.get_tensor('model.embed_tokens.weight')[:tok.get_vocab_size()].cuda().float()
    e = normalize(e-e.mean(0))
    def prototype(words):
        return torch.stack([e[w].mean(0) for w in describe(tok,words)]).mean(0)
    pos, neg, _ = SPLITS['fit']
    # Additional stress cases use the same adjectives with extra distance or negation.
    data = {k:list(rows(k)) for k in SPLITS}
    held = data['held_out']
    data['distance'] = [(s.replace("experienced ","experienced remarkably ")
                         .replace("predicting ","predicting unusually ").replace("predicted ","predicted unusually "), y)
                        for s,y in held]
    data['negated_outcome'] = [(s.replace("experienced ","experienced anything but "),-y) for s,y in held]
    path = folder/'factored.safetensors'
    result = {"revision": model.name}
    for mode in ("inherited", "permuted_vocabulary"):
        if mode == "permuted_vocabulary":
            permutation = torch.randperm(len(e), generator=torch.Generator().manual_seed(194))
            e = e[permutation]
        axes = torch.stack([prototype("received")-prototype("expected expecting"),
                            prototype(" ".join(pos))-prototype(" ".join(neg))], dim=1)
        table = (e @ axes).cpu()
        if mode == "inherited":
            save_file({"table":table},path)
            table = load_file(path)['table']
            result["bytes"] = path.stat().st_size
        suites = {}
        for key, rows_ in data.items():
            preds = [decide(table,describe(tok,s))>0 for s,_ in rows_]
            correct = [p==(y>0) for p,(_,y) in zip(preds,rows_)]
            suites[key] = {"n":len(rows_),"correct":sum(correct),
                           "both_correct":sum(a and b for a,b in zip(correct[::2],correct[1::2]))}
        result[mode] = suites
    (folder/'factored.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--compile',type=Path)
    action.add_argument('--message')
    parser.add_argument('--folder',type=Path,default=Path('artifacts/convolution_child'))
    args = parser.parse_args()
    if args.compile:
        run(args.compile,args.folder)
    else:
        tok = Tokenizer.from_file(str(args.folder/'tokenizer.json'))
        s = decide(load_file(args.folder/'factored.safetensors')['table'],describe(tok,args.message))
        print("positive" if s>0 else "negative", f"(decision score {s:+.5f})")

