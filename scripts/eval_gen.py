#!/usr/bin/env python3
"""跨生成器评测：对每个生成的 AI 文本文件算「检出率」，用留出人类文本算 FPR/整体 AUROC。

用法:
  ./env/bin/python scripts/eval_gen.py --model models/baseline \
     --gen data/generated/qwen_qa.jsonl data/generated/qwen_news.jsonl \
     --human data/processed/dev.jsonl
"""
import os, json, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    T = a.temp
    if T is None:
        tp = os.path.join(a.model, "temperature.json")
        T = json.load(open(tp))["temperature"] if os.path.exists(tp) else 1.0

    def prob(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), a.bs):
                b = tok(texts[i:i + a.bs], truncation=True, max_length=a.max_len,
                        padding=True, return_tensors="pt").to(a.device)
                lg = model(**b).logits.float() / T
                out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    human = load(a.human)
    htxt = [r["text"] for r in human if r["label"] == 0]
    ph = prob(htxt)
    print(f"人类留出样本 n={len(htxt)}  被误判为 AI 比例(FPR@0.5) = {(ph >= .5).mean():.4f}  "
          f"mean_p={ph.mean():.3f}\n")

    ai_all = []
    for gf in a.gen:
        rows = load(gf)
        if not rows:
            print(f"{os.path.basename(gf)}: 空")
            continue
        txt = [r["text"] for r in rows]
        p = prob(txt)
        ai_all.append((gf, p))
        print(f"{os.path.basename(gf):26s} n={len(rows):4d}  "
              f"检出率(p>=.5) = {(p >= .5).mean():.4f}  mean_p = {p.mean():.3f}")

    p_ai = np.concatenate([p for _, p in ai_all])
    y = np.concatenate([np.ones(len(p_ai)), np.zeros(len(ph))])
    p_all = np.concatenate([p_ai, ph])
    print(f"\n合并: AI n={len(p_ai)} 人类 n={len(ph)}")
    print(f"  AUROC(跨生成器) = {roc_auc_score(y, p_all):.4f}")
    print(f"  AI 总检出率     = {(p_ai >= .5).mean():.4f}")


if __name__ == "__main__":
    main()
