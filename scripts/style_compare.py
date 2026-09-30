#!/usr/bin/env python3
"""同域内比较「人写 vs AI 写」的风格特征均值与 Cohen's d，找出真正有效的 AI 味特征。

用法:
  ./env/bin/python scripts/style_compare.py --human data/generated/wiki_articles.jsonl \
     --ai data/generated/qwen_articles.jsonl data/generated/internlm_articles.jsonl
"""
import os, sys, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_features import extract_many, FEATURE_NAMES


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--ai", nargs="+", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    Xh = extract_many([r["text"] for r in load(a.human)])
    Xa = extract_many([r["text"] for r in load(a.ai[0])])
    for extra in a.ai[1:]:
        Xa = np.vstack([Xa, extract_many([r["text"] for r in load(extra)])])

    print(f"# {a.label}  人类 n={len(Xh)}  AI n={len(Xa)}")
    print(f"{'feature':16s} {'human':>9s} {'ai':>9s} {'cohen_d':>8s}")
    d = (Xa.mean(0) - Xh.mean(0)) / (np.sqrt((Xa.var(0) + Xh.var(0)) / 2) + 1e-9)
    for i in np.argsort(-np.abs(d)):
        print(f"{FEATURE_NAMES[i]:16s} {Xh[:,i].mean():9.3f} {Xa[:,i].mean():9.3f} {d[i]:+8.3f}")


if __name__ == "__main__":
    main()
