"""Write a rank-one adjacent-word scorer from labeled sentence reversals."""
import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from examples.convolution_child import rows
from examples.factored_child import describe, decide


def words(s):
    return re.findall("[a-z]+", s.lower())


def suites():
    fit, old = (list(rows(k)) for k in ("fit", "held_out"))
    extra = []
    for template, sign in [
        ("they promised {} food and served {} food.", -1),
        ("they served {} food after promising {} food.", 1),
        ("the meal sounded {} but tasted {}.", -1),
        ("the meal tasted {} although it sounded {}.", 1),
    ]:
        for p in ("delicious", "lovely", "tasty"):
            for n in ("disgusting", "vile", "bland"):
                extra.extend([(template.format(p,n),sign),(template.format(n,p),-sign)])
    return {"fit":fit,"previous":old,"food":extra,
            "negation":[(s.replace("experienced ","experienced anything but "),-y) for s,y in old]}


def compile_axes(embedding, tokenizer, fit, signs):
    vocab = sorted({w for s,_ in fit for w in words(s)})
    index = {w:i for i,w in enumerate(vocab)}
    vectors = torch.stack([embedding[describe(tokenizer,w)[0]].mean(0) for w in vocab]).double()
    counts = torch.zeros((len(vocab),len(vocab)),dtype=torch.float64,device=vectors.device)
    for i,(s,y) in enumerate(fit):
        tokens = words(s)
        for (a,b),n in Counter(zip(tokens,tokens[1:])).items():
            counts[index[a],index[b]] += signs[i//2]*y*n
    q,r = torch.linalg.qr(vectors.T)
    u,s,vh = torch.linalg.svd(r @ counts @ r.T)
    axes = q @ torch.stack([u[:,0],vh[0]],dim=1)
    return axes.float(),float(s[0]),float(s[1])


def verify_swap(table,tok,pairs):
    max_roundoff = 0.
    for (text,_),(swapped,_) in zip(pairs[::2],pairs[1::2]):
        a,b = words(text),words(swapped)
        i,j = [k for k,(x,y) in enumerate(zip(a,b)) if x!=y]
        assert 0<i<j-1 and j+1<len(a) and a[i+1]==a[j+1] and a[i]==b[j] and a[j]==b[i]
        def exact(w,k):
            vals = table[describe(tok,w)[0],k].tolist()
            return sum(map(Fraction,vals))/len(vals)
        def score(seq):
            return sum(exact(x,0)*exact(y,1) for x,y in zip(seq,seq[1:]))
        delta = (exact(a[i-1],0)-exact(a[j-1],0))*(exact(a[i],1)-exact(a[j],1))
        assert score(a)-score(b)==delta
        observed = decide(table,describe(tok,text))-decide(table,describe(tok,swapped))
        max_roundoff = max(max_roundoff,abs(observed-float(delta)))
    return {"rational_equalities":len(pairs)//2,"max_float32_swap_error":max_roundoff}


def run(model,folder):
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(model/"tokenizer.json",folder/"tokenizer.json")
    tok = Tokenizer.from_file(str(folder/"tokenizer.json"))
    with safe_open(model/"model.safetensors",framework="pt") as f:
        e = f.get_tensor("model.embed_tokens.weight")[:tok.get_vocab_size()].cuda().float()
    e = torch.nn.functional.normalize(e-e.mean(0))
    data = suites()
    report = {"revision":model.name,"modes":{}}
    for mode in ("inherited","shuffled_pairs","permuted_vocab"):
        signs = [1]*(len(data["fit"])//2)
        if mode=="shuffled_pairs":
            gen = torch.Generator().manual_seed(963)
            signs = [(-1 if i%2 else 1) for i in torch.randperm(len(signs),generator=gen).tolist()]
        if mode=="permuted_vocab":
            e = e[torch.randperm(len(e),generator=torch.Generator().manual_seed(194))]
        axes,first,second = compile_axes(e,tok,data["fit"],signs)
        table = (e @ axes).cpu()
        raw = [decide(table,describe(tok,s)) for s,_ in data["fit"]]
        threshold = sum(raw)/len(raw)
        if mode=="inherited":
            path = folder/"child.safetensors"
            save_file({"table":table,"threshold":torch.tensor(threshold,dtype=torch.float64)},path)
            child = load_file(path)
            table, threshold = child["table"],float(child["threshold"])
            report["child_bytes"] = path.stat().st_size
            report["swap_audit"] = verify_swap(table,tok,data["previous"])
        result = {}
        for key,suite in data.items():
            margins = [decide(table,describe(tok,s))-threshold for s,_ in suite]
            correct = [(m>0)==(y>0) for m,(_,y) in zip(margins,suite)]
            ranking = sum((p-n)*label>0 for p,n,(_,label) in zip(margins[::2],margins[1::2],suite[::2]))
            result[key] = dict(messages=len(suite),correct=sum(correct),both_correct=sum(a and b for a,b in zip(correct[::2],correct[1::2])),ordered_pairs=ranking)
        report["modes"][mode]=dict(first=first,second=second,threshold=threshold,suites=result)
    (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--compile",type=Path)
    action.add_argument("--message")
    parser.add_argument("--folder",type=Path,default=Path("artifacts/reversal_compiler"))
    args=parser.parse_args()
    if args.compile:
        run(args.compile,args.folder)
    else:
        tok=Tokenizer.from_file(str(args.folder/"tokenizer.json"))
        child=load_file(args.folder/"child.safetensors")
        margin=decide(child["table"],describe(tok,args.message))-float(child["threshold"])
        print("positive" if margin>0 else "negative",f"(margin {margin:+.6f})")
