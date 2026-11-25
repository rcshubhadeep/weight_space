"""Compose two fixed-answer LFM3 patches with a shared output direction."""
import argparse
import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from examples import lfm_boolq_benchmark as bq
from examples import lfm_sst2_edit as sst
from examples.lfm_weight_edit import Cases, Experiment


def evaluate(exp, datasets, variants, audit=False):
    report = {}
    for task, rows in datasets.items():
        totals = {name:0 for name in variants}
        paired = {name:0 for name in variants}
        audited = []
        n = 0
        for start in range(0,len(rows),256):
            group = rows[start:start+256]
            if task=="boolq":
                cases,size = bq.make_cases(group), bq.batch_size(group)
            else:
                cases,size = sst.cases(group,False), 16
            base,capture = exp.forward(cases,capture=True,answers_only=True,batch_size=size)
            n += len(base)
            for name,key in variants.items():
                scores = base if key is None else exp.score_cached_answers(capture,key,1.)
                ok = (exp.margins(scores)>0)==cases.expected_b
                totals[name] += int(ok.sum())
                if task=="boolq":
                    paired[name] += int(ok.reshape(-1,2).all(1).sum())
                if audit and name=="selected" and start in (0,256*((len(rows)-1)//256)):
                    subset = Cases(cases.prompts[:size],cases.expected_b[:size])
                    with exp.attached(key,1.):
                        live,_ = exp.forward(subset,answers_only=True,batch_size=size)
                    same = (exp.margins(live)>0)==(exp.margins(scores[:size])>0)
                    audited.append([int(same.sum()),len(same)])
            del capture
            if audit:
                print(task,min(start+256,len(rows)),flush=True)
        report[task] = dict(prompts=n,correct=totals,live_agreement=audited)
        if task=="boolq":
            report[task]["both_mappings"]=paired
    return report


def run(model,out):
    bfit,bcal,bdev,_ = bq.load_data(Path("artifacts/boolq/data"))
    _,scal,sdev = sst.splits(Path("artifacts/sst2/data"))
    exp = Experiment(model,"cuda")
    original = exp.weight_digest()
    patch = load_file("artifacts/sst2/liquid_edit/fixed_AB.safetensors",device="cuda:0")
    assert torch.equal(exp.value,patch["value"])
    bk = exp.solve(*bq.fitting_inputs(exp,bfit))*json.loads(Path("artifacts/boolq/report.json").read_text())["selected_scale"]
    sk = patch["key"]*patch["scale"]
    coefficients = list(itertools.product((0.,0.25,0.5,0.75,1.),repeat=2))
    grid = {str((a,b)):a*bk+b*sk for a,b in coefficients}
    cal = evaluate(exp,dict(boolq=bcal,sst2=scal),{"base":None,**grid})
    def objective(pair):
        fractions=[]
        for task,own in (("boolq",(1.,0.)),("sst2",(0.,1.))):
            counts=cal[task]["correct"]
            gain=counts[str(own)]-counts["base"]
            if gain<=0:
                raise ValueError("Expected both calibrated patches to improve their own task")
            fractions.append((counts[str(pair)]-counts["base"])/gain)
        return min(fractions),sum(fractions),-sum(pair)
    selected=max(coefficients,key=objective)
    variants=dict(base=None,boolq_only=bk,sst2_only=sk,sum=bk+sk,selected=grid[str(selected)])
    out.mkdir(parents=True,exist_ok=True)
    save_file({"key":variants["selected"].cpu(),"value":exp.value.cpu()},out/"combined.safetensors")
    print("Selected coefficients",selected,flush=True)
    dev=evaluate(exp,dict(boolq=bdev,sst2=sdev),variants,audit=True)
    if original!=exp.weight_digest():
        raise RuntimeError("Base matrix changed")
    result=dict(model_revision=model.name,w2_sha256=original,selected=selected,calibration=cal,validation=dev)
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dev,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("artifacts/boolq_sst2_composition"))
    args=parser.parse_args()
    run(args.model,args.out)
