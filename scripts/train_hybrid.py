#!/usr/bin/env python3
"""混合检测器：中文编码器（CLS）+ 显式「AI 味」风格特征 → MLP 分类头。

风格特征直击用户关注点（破折号/冒号/括号名词解释/表格/比喻/套话/句式节奏）。
支持消融：--no_style（纯编码器）对比。

用法:
  ./env/bin/python scripts/train_hybrid.py --data_dir data/processed_v2 --out models/hybrid_v2
"""
import os, sys, json, argparse, math, contextlib, numpy as np, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


class Hybrid(nn.Module):
    def __init__(self, base, n_style, hidden=256, num_labels=2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.Linear(h + n_style, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, num_labels))

    def forward(self, input_ids, attention_mask, style):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.drop(torch.cat([cls, style], dim=-1)))


def batches(ids, style, labels, bs, shuffle=True, max_len=512):
    idx = np.random.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        seqs = [ids[j][:max_len] for j in b]
        L = max(len(s) for s in seqs)
        inp = torch.zeros(len(b), L, dtype=torch.long)
        am = torch.zeros(len(b), L, dtype=torch.long)
        for r, s in enumerate(seqs):
            inp[r, :len(s)] = torch.tensor(s)
            am[r, :len(s)] = 1
        yield inp, am, style[b], labels[b]


def ac(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()


def evaluate(model, ids, style, labels, dev):
    model.eval()
    ps = []
    with torch.no_grad(), ac(dev):
        for inp, am, st, _ in batches(ids, style, labels, 64, shuffle=False):
            ps.append(torch.softmax(model(inp.to(dev), am.to(dev), st.to(dev)).float(), -1)[:, 1].cpu())
    p = torch.cat(ps).numpy()
    y = labels.numpy()
    out = dict(acc=accuracy_score(y, (p >= .5).astype(int)),
               f1=f1_score(y, (p >= .5).astype(int), average="macro"))
    try:
        out["auroc"] = roc_auc_score(y, p)
    except ValueError:
        out["auroc"] = float("nan")
    conf = np.maximum(p, 1 - p); corr = (p >= .5).astype(int) == y
    e = 0
    for i in range(10):
        m = (conf > i / 10) & (conf <= (i + 1) / 10)
        if m.sum():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    out["ece"] = e
    return out, p


def fit_temp(logits, labels):
    x = torch.tensor(logits, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1)); opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    lf = nn.CrossEntropyLoss()
    def cl():
        opt.zero_grad(); l = lf(x / T.clamp(min=0.05), y); l.backward(); return l
    opt.step(cl); return float(T.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed_v2"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "hybrid_v2"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--no_style", action="store_true")
    ap.add_argument("--features", default="style+logic", choices=["style", "style+logic"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu"

    tok = AutoTokenizer.from_pretrained(a.base)
    tr = load(os.path.join(a.data_dir, "train.jsonl"))
    dv = load(os.path.join(a.data_dir, "dev.jsonl"))
    if a.smoke:
        tr, dv = tr[:1500], dv[:400]

    def prep(rows):
        ids = [tok(r["text"], truncation=True, max_length=a.max_len)["input_ids"] for r in rows]
        st = FEAT.extract_many([r["text"] for r in rows], a.features)
        y = np.array([r["label"] for r in rows], dtype=np.int64)
        return ids, st, torch.tensor(y)

    ids_tr, st_tr, y_tr = prep(tr)
    ids_dv, st_dv, y_dv = prep(dv)
    mu, sd = st_tr.mean(0), st_tr.std(0) + 1e-6
    if a.no_style:
        st_tr = np.zeros((len(st_tr), 0), dtype=np.float32)
        st_dv = np.zeros((len(st_dv), 0), dtype=np.float32)
    else:
        st_tr = ((st_tr - mu) / sd).astype(np.float32)
        st_dv = ((st_dv - mu) / sd).astype(np.float32)

    model = Hybrid(a.base, st_tr.shape[1]).to(dev)
    print(f"style dims = {st_tr.shape[1]}  ({'纯编码器' if a.no_style else '混合'})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(tr) / a.bs) * a.epochs
    warm = int(0.1 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
                                              (1 - s / steps) if s < steps else 0.0)

    step = 0
    for ep in range(int(a.epochs)):
        model.train()
        for inp, am, stb, yb in batches(ids_tr, torch.tensor(st_tr), y_tr, a.bs):
            inp, am, stb, yb = inp.to(dev), am.to(dev), stb.to(dev), yb.to(dev)
            with ac(dev):
                logits = model(inp, am, stb)
                loss = nn.CrossEntropyLoss()(logits.float(), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"  ep{ep} step{step} loss={loss.item():.4f}", flush=True)
        m, _ = evaluate(model, ids_dv, torch.tensor(st_dv), y_dv, dev)
        print(f"epoch {ep}: dev {m}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump(dict(base=a.base, max_len=a.max_len, no_style=a.no_style,
                   feature_kind=a.features,
                   style_mean=mu.tolist() if not a.no_style else [],
                   style_std=sd.tolist() if not a.no_style else [],
                   feature_names=FEAT.names(a.features)), open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
