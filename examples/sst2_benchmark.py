"""Compare a frozen reversal child and SST-2-trained standalone classifiers."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
import torch

from examples.factored_child import decide, describe
from examples.family import Teacher, Task


def lexical_terms(tokens):
    return {str(i) for i in tokens} | {f"{a},{b}" for a,b in zip(tokens,tokens[1:])}


def fit_lexical(encoded, labels):
    counts = [Counter(), Counter()]
    totals = [0,0]
    for tokens,y in zip(encoded,labels):
        counts[y].update(lexical_terms(tokens))
        totals[y] += 1
    return {term:math.log((counts[1][term]+1)/(totals[1]+2))
                 -math.log((counts[0][term]+1)/(totals[0]+2))
            for term in counts[0].keys() | counts[1].keys()}


def lexical_score(weights, seq):
    return sum(weights.get(term,0.) for term in sorted(lexical_terms(seq)))/max(1,len(seq))**0.5


def sentence_vectors(e, seqs):
    parts = []
    for start in range(0,len(seqs),128):
        batch = seqs[start:start+128]
        ids = torch.tensor([i for row in batch for i in row],device=e.device)
        offsets = torch.tensor(np.cumsum([0]+[len(row) for row in batch[:-1]]),device=e.device)
        parts.append(torch.nn.functional.embedding_bag(ids,e,offsets,mode="mean"))
    return torch.cat(parts)


def encode(tok,texts):
    return [r.ids for r in tok.encode_batch([" "+s.strip().lower() for s in texts],add_special_tokens=False)]


def threshold(scores,labels):
    order = np.argsort(scores,kind="stable")
    values = scores[order]
    signs = 1-2*labels[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(values))+1,len(values)]
    best = cuts[np.argmax(np.r_[0,signs.cumsum()][cuts])]
    if best==0:
        return float(np.nextafter(values[0],-np.inf))
    return float(values[best-1])


def accuracy(scores,t,y):
    return int(np.sum((scores>t)==y))


def timed(fn,texts):
    fn(texts[0])
    elapsed = []
    for s in texts[:64]:
        start = perf_counter()
        fn(s)
        elapsed.append((perf_counter()-start)*1000)
    return median(elapsed)


def load_child(out):
    stored = json.loads((out/"classifier.json").read_text())
    table = load_file(out/"table.safetensors")["semantic"]
    tok = Tokenizer.from_file(str(out/"tokenizer.json"))
    def predict(s):
        seq = encode(tok,[s])[0]
        c = stored["config"]
        return ((1-c["mix"])*float(table[seq].mean())/c["semantic_scale"]
                +c["mix"]*lexical_score(stored["lexical"],seq)/c["lexical_scale"]-c["threshold"])
    return predict


def run(model,data,prior,out):
    out.mkdir(parents=True,exist_ok=True)
    trainfile,devfile = [next(data.glob(split+"*.parquet")) for split in ("train","validation")]
    train = pq.read_table(trainfile).to_pylist()
    dev = pq.read_table(devfile).to_pylist()
    if (len(train),len(dev))!=(67349,872):
        raise ValueError("Expected complete SST-2 train and validation splits")
    def canonical(s):
        return " ".join(s.lower().split())
    used = {canonical(r["sentence"]) for r in dev}
    unique = []
    for i in np.random.default_rng(7331).permutation(len(train)):
        r = train[i]
        key = canonical(r["sentence"])
        if key and key not in used:
            unique.append(r)
            used.add(key)
    cal,fit = unique[:1024],unique[1024:]
    tok = Tokenizer.from_file(str(model/"tokenizer.json"))
    texts = [r["sentence"] for r in fit+cal]
    encoded = encode(tok,texts)
    labels = np.array([r["label"] for r in fit+cal])
    n = len(fit)
    lex = fit_lexical(encoded[:n],labels[:n])
    lx = np.array([lexical_score(lex,s) for s in encoded])
    lex_scale = max(lx[:n].std(),1e-8)
    lt = threshold(lx[n:],labels[n:])
    with safe_open(model/"model.safetensors",framework="pt") as f:
        e = f.get_tensor("model.embed_tokens.weight")[:tok.get_vocab_size()].cuda().float()
    e = torch.nn.functional.normalize(e-e.mean(0))
    x = sentence_vectors(e,encoded)
    center = x[:n].mean(0)
    design = x[:n]-center
    y = torch.tensor(labels[:n]*2.-1.,device=e.device,dtype=x.dtype)
    gram = design.T @ design
    rhs = design.T @ (y-y.mean())
    trials = []
    best = None
    for ridge in (0.01,0.1,1.0):
        w = torch.linalg.solve(gram+ridge*torch.eye(len(center),device=e.device),rhs)
        table = (e @ w).cpu()
        # Use the deployed lookup calculation on calibration too.
        sm = np.array([float(table[s].mean()) for s in encoded])
        scale = max(sm[:n].std(),1e-8)
        for mix in (0.,0.25,0.5,0.75,1.):
            combined = (1-mix)*sm[n:]/scale+mix*lx[n:]/lex_scale
            t = threshold(combined,labels[n:])
            correct = accuracy(combined,t,labels[n:])
            trials.append(dict(ridge=ridge,mix=mix,correct=correct))
            if best is None or correct>best[0]:
                best = correct,table.clone(),dict(mix=mix,ridge=ridge,threshold=t,
                                                 lexical_scale=lex_scale,semantic_scale=scale)
    _,table,config = best
    save_file({"semantic":table},out/"table.safetensors")
    (out/"classifier.json").write_text(json.dumps({"config":config,"lexical":lex},sort_keys=True))
    (out/"tokenizer.json").write_bytes((model/"tokenizer.json").read_bytes())
    del e,x,design,gram
    torch.cuda.empty_cache()
    child = load_child(out)
    def baseline(s):
        return lexical_score(lex,encode(tok,[s])[0])-lt
    previous = load_file(prior/"child.safetensors")
    oldtok = Tokenizer.from_file(str(prior/"tokenizer.json"))
    def old(s):
        return decide(previous["table"],describe(oldtok,s))-float(previous["threshold"])
    texts = [r["sentence"] for r in dev]
    gold = np.array([r["label"] for r in dev])
    predictions = {}
    report = {"model_revision":model.name,"input_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (trainfile,devfile)},"fit":n,"calibration":len(cal),"validation":len(dev),
              "calibration_trials":trials,"selected":config,"systems":{}}
    for name,fn,files in (("compiled",child,[out/"table.safetensors",out/"classifier.json",out/"tokenizer.json"]),
                          ("frozen_reversal",old,[prior/"child.safetensors",prior/"tokenizer.json"]),
                          ("lexical",baseline,[])):
        scores = np.array([fn(s) for s in texts])
        predictions[name] = scores>0
        report["systems"][name] = dict(correct=accuracy(scores,0,gold),median_ms=timed(fn,texts))
        if files:
            report["systems"][name]["bytes_with_tokenizer"]=sum(p.stat().st_size for p in files)
    baseline_file = out/"lexical_baseline.json"
    baseline_file.write_text(json.dumps({"weights":lex,"threshold":lt},sort_keys=True))
    report["systems"]["lexical"]["bytes_with_tokenizer"] = baseline_file.stat().st_size+(out/"tokenizer.json").stat().st_size
    parent = Teacher(model,"cuda")
    task = Task("What is the sentiment expressed in this movie review?",("negative","positive"),(),(),())
    scores = parent.predict(texts,task).numpy()
    report["systems"]["parent_fixed_AB"] = dict(correct=accuracy(scores,0,gold),
        median_ms=timed(lambda s: parent.predict([s],task),texts),
        checkpoint_bytes=(model/"model.safetensors").stat().st_size+(model/"tokenizer.json").stat().st_size)
    predictions["parent_fixed_AB"] = scores>0
    compiled = predictions["compiled"]==gold
    report["paired_vs_compiled"] = {}
    for name,pred in predictions.items():
        other = pred==gold
        report["paired_vs_compiled"][name] = dict(compiled_only=int((compiled & ~other).sum()),
                                                  other_only=int((other & ~compiled).sum()))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["systems"],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--model",type=Path)
    action.add_argument("--message")
    parser.add_argument("--data",type=Path,default=Path("artifacts/sst2/data"))
    parser.add_argument("--prior",type=Path,default=Path("artifacts/reversal_compiler"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/sst2/benchmark"))
    args=parser.parse_args()
    if args.model:
        run(args.model,args.data,args.prior,args.out)
    else:
        margin = load_child(args.out)(args.message)
        print("positive" if margin>0 else "negative", f"(margin {margin:+.4f})")
