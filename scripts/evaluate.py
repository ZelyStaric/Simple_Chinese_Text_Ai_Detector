#!/usr/bin/env python3
"""评测检测器：acc/f1/auroc/ece + 可靠性表，可套温度校准。

用法:
  ./env/bin/python scripts/evaluate.py --model models/baseline --data data/processed/dev.jsonl
"""
import os, json, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def load_jsonl(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows


def ece(probs, labels, n_bins=10):
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5).astype(int) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def reliability(probs, labels, n_bins=10):
    conf = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5).astype(int) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        rows.append((f"{bins[i]:.1f}-{bins[i+1]:.1f}", int(m.sum()),
                     float(conf[m].mean()), float(correct[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--temp", type=float, default=None, help="不传则读 model/temperature.json")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()

    T = a.temp
    if T is None:
        tp = os.path.join(a.model, "temperature.json")
        if os.path.exists(tp):
            T = json.load(open(tp))["temperature"]
    T = T or 1.0

    rows = load_jsonl(a.data)
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows])

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), a.bs):
            b = tok(texts[i:i + a.bs], truncation=True, max_length=a.max_len,
                    padding=True, return_tensors="pt").to(a.device)
            logits = model(**b).logits.float() / T
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(int)

    print(f"file={os.path.basename(a.data)}  n={len(rows)}  T={T:.4f}")
    print(f"  acc   = {accuracy_score(labels, pred):.4f}")
    print(f"  f1    = {f1_score(labels, pred, average='macro'):.4f}")
    print(f"  auroc = {roc_auc_score(labels, p):.4f}")
    print(f"  ece   = {ece(p, labels):.4f}")
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("  reliability (bin, n, mean_conf, acc):")
    for r in reliability(p, labels):
        print(f"    {r[0]:>9s}  n={r[1]:6d}  conf={r[2]:.3f}  acc={r[3]:.3f}")


if __name__ == "__main__":
    main()
