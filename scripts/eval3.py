#!/usr/bin/env python3
"""三类模型评测：dev / 留出域 / 留出生成器(分域) / 博客打分。"""
import os, sys, json, argparse, numpy as np, torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT
from train_hybrid import Hybrid, ac
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["human", "AI", "AI-edited"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def predict(model, tok, rows, dev, meta):
    ids = [tok(r["text"], truncation=True, max_length=meta["max_len"])["input_ids"] for r in rows]
    st = FEAT.extract_many([r["text"] for r in rows], meta["feature_kind"])
    st = ((st - np.array(meta["style_mean"])) / np.array(meta["style_std"])).astype(np.float32)
    P = []
    with torch.no_grad(), ac(dev):
        for i in range(0, len(rows), 64):
            seqs = [s[:meta["max_len"]] for s in ids[i:i + 64]]
            L = max(len(s) for s in seqs)
            inp = torch.zeros(len(seqs), L, dtype=torch.long); am = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            lg = model(inp.to(dev), am.to(dev), torch.tensor(st[i:i + 64]).to(dev))
            P.append(torch.softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed_v4"))
    ap.add_argument("--text", default=os.path.join(ROOT, "data", "blog", "zelystaric_rdna3.txt"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.model, "meta.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Hybrid(meta["base"], len(meta["style_mean"]), num_labels=3).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev)); model.eval()

    for f in ["dev.jsonl", "test_unseen_dom.jsonl"]:
        p = os.path.join(a.data_dir, f)
        if not os.path.exists(p):
            continue
        rows = load(p); P = predict(model, tok, rows, dev, meta)
        y = np.array([r["label"] for r in rows]); pred = P.argmax(1)
        print(f"{f:24s} n={len(rows)} 3cls acc={accuracy_score(y,pred):.4f} macroF1={f1_score(y,pred,average='macro'):.4f}")

    rows = load(os.path.join(a.data_dir, "test_unseen_gen.jsonl"))
    P = predict(model, tok, rows, dev, meta); pred = P.argmax(1)
    y = np.array([r["label"] for r in rows])
    print(f"\n留出生成器(Qwen)  n={len(rows)} 3cls acc={accuracy_score(y,pred):.4f}")
    print("  cm(真 0/1/2 × 预测):"); print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    doms = {}
    for r, pr in zip(rows, pred):
        doms.setdefault(r["domain"], []).append((r["label"], pr))
    print(f"\n  {'domain':16s} {'n':>5s} {'真label':>8s} {'命中(正确类)':>12s}")
    for d, items in sorted(doms.items(), key=lambda x: -len(x[1])):
        arr = np.array(items)
        print(f"  {d:16s} {len(arr):5d} {dict((int(k),int(v)) for k,v in zip(*np.unique(arr[:,0],return_counts=True)))}  "
              f"acc={(arr[:,0]==arr[:,1]).mean():.3f}")

    # 博客多窗口打分（按字符切窗并重新 tokenize，保留特殊符号）
    txt = open(a.text, encoding="utf-8").read()
    W = meta["max_len"]; chars = max(400, W * 2); step = max(200, chars // 2)
    wins = [txt[i:i + chars] for i in range(0, max(1, len(txt)), step)]
    wins = [w for w in wins if w.strip()]
    st_row = ((FEAT.extract(txt, meta["feature_kind"]) - np.array(meta["style_mean"])) /
              np.array(meta["style_std"])).astype(np.float32)
    probs = []
    with torch.no_grad(), ac(dev):
        for w in wins:
            ids = tok(w, truncation=True, max_length=W)["input_ids"]
            lg = model(torch.tensor([ids]).to(dev), torch.ones(1, len(ids), dtype=torch.long).to(dev),
                       torch.tensor(st_row)[None].to(dev))
            probs.append(torch.softmax(lg.float(), -1)[0].cpu().numpy())
    Pm = np.mean(probs, axis=0)
    print(f"\n博客 «{os.path.basename(a.text)}» ({len(wins)} 窗口)  平均概率: "
          f"human={Pm[0]:.3f}  AI={Pm[1]:.3f}  AI-edited={Pm[2]:.3f}  -> {NAMES[int(Pm.argmax())]}")


if __name__ == "__main__":
    main()
