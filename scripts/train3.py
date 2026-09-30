#!/usr/bin/env python3
"""三类训练：human(0) / AI(1) / AI-edited(2)。用法同 train_hybrid。"""
import os, sys, json, argparse, math, contextlib, numpy as np, torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT
from train_hybrid import Hybrid, ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def batches(ids, style, labels, bs, shuffle=True, max_len=512):
    idx = np.random.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        seqs = [ids[j][:max_len] for j in b]
        L = max(len(s) for s in seqs)
        inp = torch.zeros(len(b), L, dtype=torch.long); am = torch.zeros(len(b), L, dtype=torch.long)
        for r, s in enumerate(seqs):
            inp[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        yield inp, am, style[b], labels[b]


def evaluate(model, ids, style, labels, dev):
    model.eval(); ps = []
    with torch.no_grad(), ac(dev):
        for inp, am, st, _ in batches(ids, style, labels, 64, shuffle=False):
            ps.append(torch.softmax(model(inp.to(dev), am.to(dev), st.to(dev)).float(), -1).cpu())
    P = torch.cat(ps).numpy(); y = labels.numpy(); pred = P.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro")), P, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed_v4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "hybrid3_v4"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--features", default="style+logic")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.base)
    tr = load(os.path.join(a.data_dir, "train.jsonl"))
    dv = load(os.path.join(a.data_dir, "dev.jsonl"))

    def prep(rows):
        ids = [tok(r["text"], truncation=True, max_length=a.max_len)["input_ids"] for r in rows]
        st = FEAT.extract_many([r["text"] for r in rows], a.features)
        y = np.array([r["label"] for r in rows], dtype=np.int64)
        return ids, st, torch.tensor(y)

    ids_tr, st_tr, y_tr = prep(tr)
    ids_dv, st_dv, y_dv = prep(dv)
    mu, sd = st_tr.mean(0), st_tr.std(0) + 1e-6
    st_tr = ((st_tr - mu) / sd).astype(np.float32); st_dv = ((st_dv - mu) / sd).astype(np.float32)

    model = Hybrid(a.base, st_tr.shape[1], num_labels=3).to(dev)
    cnt = np.bincount(y_tr.numpy(), minlength=3).astype(np.float32)
    cw = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    print("class counts", cnt.tolist(), "weights", [round(float(x), 2) for x in cw])
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(tr) / a.bs) * a.epochs; warm = int(0.1 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * (1 - s / steps) if s < steps else 0.0)

    step = 0
    for ep in range(int(a.epochs)):
        model.train()
        for inp, am, stb, yb in batches(ids_tr, torch.tensor(st_tr), y_tr, a.bs):
            inp, am, stb, yb = inp.to(dev), am.to(dev), stb.to(dev), yb.to(dev)
            with ac(dev):
                loss = crit(model(inp, am, stb).float(), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"  ep{ep} step{step} loss={loss.item():.4f}", flush=True)
        m, P, cm = evaluate(model, ids_dv, torch.tensor(st_dv), y_dv, dev)
        print(f"epoch {ep}: dev {m}\n  cm(rows=true 0/1/2, cols=pred):\n{cm}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump(dict(base=a.base, max_len=a.max_len, feature_kind=a.features, num_labels=3,
                   style_mean=mu.tolist(), style_std=sd.tolist(),
                   feature_names=FEAT.names(a.features)), open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
