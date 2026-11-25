"""Extract first-layer short-convolution channels and solve a finite margin game."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer


SPLITS = {
    "fit": (("good", "great", "excellent"), ("bad", "poor", "terrible"),
            ("i expected {} service but received {} service.", "i received {} service despite expecting {} service.")),
    "held_out": (("wonderful", "superb", "fantastic", "outstanding"),
                 ("awful", "horrible", "dreadful", "abysmal"),
                 ("i predicted {} service but experienced {} service.", "i experienced {} service despite predicting {} service.")),
}


def rows(split):
    pos, neg, templates = SPLITS[split]
    for index, template in enumerate(templates):
        for p in pos:
            for n in neg:
                for a, b, label in ((p, n, -1), (n, p, 1)):
                    yield template.format(a, b), label if index == 0 else -label


def features(table, kernel, ids):
    bx, c = table[ids].chunk(2, dim=1)
    windows = torch.nn.functional.pad(bx.T, (kernel.shape[1]-1, 0)).unfold(1, kernel.shape[1], 1)
    return ((windows * kernel[:, None]).sum(-1) * c.T).mean(1)


def game(x, labels):
    # Positive matrix game. Solve max sum(y) subject to A.T y <= 1 by simplex.
    signed = np.asarray(labels)[:, None] * x
    a = 2 + np.concatenate([signed, -signed], axis=1)
    m, n = a.T.shape
    t = np.zeros((m+1, n+m+1))
    t[:m, :n], t[:m, n:-1], t[:m, -1], t[-1, :n] = a.T, np.eye(m), 1, -1
    basis = np.arange(n, n+m)
    for _ in range(20000):
        candidates = np.flatnonzero(t[-1, :-1] < -1e-10)
        if not len(candidates):
            break
        col = candidates[0]
        valid = np.flatnonzero(t[:-1, col] > 1e-10)
        ratios = t[valid, -1] / t[valid, col]
        tied = valid[ratios <= ratios.min()+1e-12]
        row = tied[basis[tied].argmin()]
        pivot = t[row].copy() / t[row, col]
        t -= t[:, col, None] * pivot
        t[row] = pivot
        basis[row] = col
    else:
        raise RuntimeError("Simplex iteration limit")
    probs = t[-1, n:-1] / t[-1, -1]
    dual = np.zeros(n+m)
    dual[basis] = t[:-1,-1] / t[-1,-1]
    upper = (dual[:n] @ a).max()
    lower = (a @ probs).min()
    if (abs(upper-lower)>1e-7 or min(probs.min(),dual.min()) < -1e-8
            or abs(probs.sum()-1)>1e-8 or abs(dual[:n].sum()-1)>1e-8):
        raise RuntimeError("Primal/dual check failed")
    left, right = np.split(probs, 2)
    return left-right, float(lower-2), float(upper-lower)


def readout(f, center, spread):
    return ((f-center)/spread).clamp(-1,1)


def run(model, out):
    out.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(model/'tokenizer.json', out/'tokenizer.json')
    tok = Tokenizer.from_file(str(out/'tokenizer.json'))
    all_rows = [*rows("fit"), *rows("held_out")]
    sequences = [tok.encode(s, add_special_tokens=False).ids for s,_ in all_rows]
    labels = np.array([y for _,y in all_rows])
    count = len(list(rows("fit")))
    vocab = tok.get_vocab_size()
    unique = torch.tensor(sorted(set(sum(sequences, []))),device='cuda')
    with safe_open(model/'model.safetensors', framework='pt') as f:
        emb = f.get_tensor('model.embed_tokens.weight')[:vocab].cuda().float()
        prefix = 'model.layers.0.'
        gain = f.get_tensor(prefix+'operator_norm.weight').cuda().float()
        proj = f.get_tensor(prefix+'conv.in_proj.weight').cuda().float().reshape(3,-1,emb.shape[1])
        kernel = f.get_tensor(prefix+'conv.conv.weight').cuda().float().squeeze(1)
    def lookup(indices, channels):
        e = emb[indices]
        e = e * torch.rsqrt(e.square().mean(1,keepdim=True)+1e-5) * gain
        b,c,x = (e @ w[channels].T for w in proj)
        return torch.cat([b*x,c],dim=1)

    seq = [torch.searchsorted(unique,torch.tensor(s,device='cuda')) for s in sequences]
    all_ch = torch.arange(kernel.shape[0],device='cuda')
    report = {"revision":model.name,"fit_messages":count,"held_out_messages":len(seq)-count,"runs":[]}
    for mode in ("inherited","permuted_vocabulary","current_token_only"):
        lookup_ids = unique
        if mode == "permuted_vocabulary":
            lookup_ids = torch.randperm(vocab,generator=torch.Generator().manual_seed(194))[unique.cpu()].cuda()
        k = kernel.clone()
        if mode == "current_token_only":
            k[:,:-1] = 0
        table = lookup(lookup_ids,all_ch)
        raw = torch.stack([features(table,k,s) for s in seq]).cpu().double()
        center = raw[:count].mean(0)
        spread = (raw[:count]-center).abs().amax(0).clamp_min(1e-12)
        normalized = readout(raw,center,spread)
        rank = (normalized[:count].T @ torch.tensor(labels[:count]).double()).abs().argsort(descending=True)
        for width in (1,4,16):
            channels = rank[:width]
            x = np.column_stack([normalized[:,channels].numpy(),np.ones(len(seq))])
            weights,margin,gap = game(x[:count],labels[:count])
            prediction = np.where(x @ weights > 0,1,-1)
            correct = prediction[count:]==labels[count:]
            record = dict(mode=mode,width=width,fit_margin=margin,duality_gap=gap,
                          fit_correct=int(sum(prediction[:count]==labels[:count])),
                          held_out_correct=int(correct.sum()),both_correct=int(correct.reshape(-1,2).all(1).sum()))
            report["runs"].append(record)
            if mode=="inherited" and width==16:
                ch = channels.cuda()
                full = torch.cat([lookup(i,ch).cpu() for i in torch.arange(vocab,device='cuda').split(2048)])
                path = out/'child.safetensors'
                save_file(dict(table=full,kernel=k[ch].cpu(),center=center[channels],spread=spread[channels],
                               weights=torch.from_numpy(weights),channels=channels),path)
                child = load_file(path)
                scores = [score(child,s) for s in sequences]
                if not np.array_equal(np.where(np.array(scores)>0,1,-1),prediction):
                    raise RuntimeError("Exported child disagrees with compiler")
                report["child_bytes"] = path.stat().st_size
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def score(child, ids):
    f = features(child['table'],child['kernel'],ids).double()
    x = readout(f,child['center'],child['spread'])
    w = child['weights']
    return float(x @ w[:-1] + w[-1])


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
        tokenizer = Tokenizer.from_file(str(args.folder/'tokenizer.json'))
        tokens = tokenizer.encode(args.message.strip().lower(),add_special_tokens=False).ids
        if not tokens:
            parser.error("Enter a nonempty message")
        value = score(load_file(args.folder/'child.safetensors'), tokens)
        print("positive" if value>0 else "negative", f"(decision score {value:+.5f})")
