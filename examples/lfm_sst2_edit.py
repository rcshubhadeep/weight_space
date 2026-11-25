"""Fit and evaluate removable final-layer LFM edits on SST-2."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from torch.nn.functional import normalize

from examples.lfm_weight_edit import Cases, Experiment
from examples.sst2_benchmark import threshold


def cases(rows, native):
    instructions = "What is the sentiment expressed in this movie review? "
    instructions += ("Choose only negative or positive." if native else
                     "A = negative. B = positive. Choose only A or B.")
    return Cases([instructions+"\n\n"+r["sentence"] for r in rows],
                 torch.tensor([r["label"] for r in rows]))


def splits(data):
    train, dev = (pq.read_table(next(data.glob(name+"*.parquet"))).to_pylist()
                  for name in ("train","validation"))
    assert (len(train),len(dev))==(67349,872)
    used = {" ".join(r["sentence"].lower().split()) for r in dev}
    sample = []
    for i in np.random.default_rng(1021).permutation(len(train)):
        row = train[i]
        key = " ".join(row["sentence"].lower().split())
        if key and key not in used:
            sample.append(row)
            used.add(key)
        if len(sample)==768:
            break
    return sample[:512],sample[512:],dev


def run(model,data,out):
    out.mkdir(exist_ok=True,parents=True)
    fit_rows,cal_rows,dev_rows = splits(data)
    exp = Experiment(model,"cuda")
    digest = exp.weight_digest()
    results = {}
    for name in ("fixed_AB","native"):
        native = name=="native"
        if native:
            ids = [exp.teacher.tokenizer.encode(s,add_special_tokens=False) for s in (" negative"," positive")]
            if any(len(x)!=1 for x in ids):
                raise ValueError(f"Expected single-token sentiment words, got {ids}")
            exp.answer_rows = exp.model.get_output_embeddings().weight.detach()[[x[0] for x in ids]]
            exp.value = normalize((exp.answer_rows[1]-exp.answer_rows[0]).float(),dim=0)
        fit,cal,dev = [cases(r,native) for r in (fit_rows,cal_rows,dev_rows)]
        base,cap = exp.forward(fit,answers_only=True,capture=True)
        key = exp.solve(fit,base,cap)
        del cap
        base,cap = exp.forward(cal,answers_only=True,capture=True)
        cutoff = threshold(exp.margins(base).numpy(),cal.expected_b.numpy())
        scales = (0.,0.125,0.25,0.5,1.,2.)
        trials = []
        for scale in scales:
            scores = exp.score_cached_answers(cap,key,scale) if scale else base
            trials.append(int(((exp.margins(scores)>0)==cal.expected_b).sum()))
        chosen = scales[np.argmax(trials)]
        del cap
        base,_ = exp.forward(dev,answers_only=True)
        with exp.attached(key,chosen):
            edited,_ = exp.forward(dev,answers_only=True)
        gold = dev.expected_b
        before = (exp.margins(base)>0)==gold
        after = (exp.margins(edited)>0)==gold
        bias = (exp.margins(base)>cutoff)==gold
        report = dict(calibration_correct=trials,scales=scales,selected=chosen,
                      base=int(before.sum()),edited=int(after.sum()),fixed=int((~before & after).sum()),
                      broken=int((before & ~after).sum()),threshold_only=int(bias.sum()),
                      threshold=float(cutoff))
        save_file({"key":key.cpu(),"value":exp.value.cpu(),"scale":torch.tensor(chosen)},out/(name+".safetensors"))
        results[name]=report
        print(name,report,flush=True)
    if digest!=exp.weight_digest():
        raise RuntimeError("Base W2 digest changed")
    report=dict(model_revision=model.name,fit=len(fit_rows),calibration=len(cal_rows),validation=len(dev_rows),
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.parquet"))},
                w2_sha256=digest,base_w2_unchanged=True,systems=results)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


def replay(model,data,out):
    """Replay the saved branches through the ordinary full-vocabulary output head."""
    path = out/"report.json"
    report = json.loads(path.read_text())
    if model.name != report["model_revision"]:
        raise ValueError("Checkpoint revision differs from the fitted branch")
    _,_,rows = splits(data)
    exp = Experiment(model,"cuda")
    if exp.weight_digest()!=report["w2_sha256"]:
        raise ValueError("Unexpected W2 matrix")
    for name,result in report["systems"].items():
        native = name=="native"
        inputs = cases(rows,native)
        patch = load_file(out/(name+".safetensors"),device="cuda:0")
        labels = (" negative"," positive") if native else (" A"," B")
        ids = [exp.teacher.tokenizer.encode(s,add_special_tokens=False)[0] for s in labels]
        exp.value = patch["value"]
        def evaluate():
            scores,_ = exp.forward(inputs,batch_size=12)
            margins = scores[:,ids[1]]-scores[:,ids[0]]
            return margins
        base = evaluate()
        with exp.attached(patch["key"],float(patch["scale"])):
            edited = evaluate()
        before = (base>0)==inputs.expected_b
        after = (edited>0)==inputs.expected_b
        result["full_head_replay"] = dict(base=int(before.sum()),edited=int(after.sum()),
                threshold_only=int(((base>result["threshold"])==inputs.expected_b).sum()),
                fixed=int((~before & after).sum()),broken=int((before & ~after).sum()))
        print(name,result["full_head_replay"],flush=True)
    path.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--data",type=Path,default=Path("artifacts/sst2/data"))
    p.add_argument("--out",type=Path,default=Path("artifacts/sst2/liquid_edit"))
    p.add_argument("--replay",action="store_true")
    args=p.parse_args()
    if args.replay:
        replay(args.model,args.data,args.out)
    else:
        run(args.model,args.data,args.out)
