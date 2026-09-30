#!/usr/bin/env python3
"""用「表层风格特征」训逻辑回归，看这些 AI 味特征单独能判多准（可解释）。

用法:
  ./env/bin/python scripts/style_model.py --gen data/generated/qwen_qa.jsonl ...
"""
import os, json, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from style_features import extract_many, FEATURE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--gen", nargs="*", default=[])
    a = ap.parse_args()

    tr = load(os.path.join(a.data_dir, "train.jsonl"))
    dv = load(os.path.join(a.data_dir, "dev.jsonl"))
    Xtr = extract_many([r["text"] for r in tr]); ytr = np.array([r["label"] for r in tr])
    Xdv = extract_many([r["text"] for r in dv]); ydv = np.array([r["label"] for r in dv])

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)

    pdv = clf.predict_proba(sc.transform(Xdv))[:, 1]
    print(f"dev:            acc={accuracy_score(ydv,(pdv>=.5)):.4f}  auroc={roc_auc_score(ydv,pdv):.4f}")

    dom = load(os.path.join(a.data_dir, "test_unseen_dom.jsonl"))
    Xd = extract_many([r["text"] for r in dom]); ydom = np.array([r["label"] for r in dom])
    pd = clf.predict_proba(sc.transform(Xd))[:, 1]
    print(f"unseen_domain:  acc={accuracy_score(ydom,(pd>=.5)):.4f}  auroc={roc_auc_score(ydom,pd):.4f}")

    ph = clf.predict_proba(sc.transform(Xdv[ydv == 0]))[:, 1]
    print(f"\n人类留出 FPR@0.5 = {(ph >= .5).mean():.4f}")
    allp = []
    for gf in a.gen:
        rows = load(gf)
        if not rows:
            continue
        p = clf.predict_proba(sc.transform(extract_many([r["text"] for r in rows])))[:, 1]
        allp.append(p)
        print(f"{os.path.basename(gf):22s} n={len(rows):4d} 检出率={(p>=.5).mean():.4f} mean_p={p.mean():.3f}")
    if allp:
        pai = np.concatenate(allp)
        y = np.concatenate([np.ones(len(pai)), np.zeros(len(ph))])
        pa = np.concatenate([pai, ph])
        print(f"跨生成器合并 AUROC = {roc_auc_score(y, pa):.4f}")

    coef = clf.coef_[0]
    order = np.argsort(-np.abs(coef))
    print("\n=== 最有区分力的风格特征（标准化系数，+ 表示更像 AI）===")
    for i in order[:15]:
        print(f"  {FEATURE_NAMES[i]:16s} {coef[i]:+.3f}")


if __name__ == "__main__":
    main()
