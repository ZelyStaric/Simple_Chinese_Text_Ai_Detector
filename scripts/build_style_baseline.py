#!/usr/bin/env python3
"""从真人语料统计 43 维风格特征的均值/方差，供「AI 味报告」做 z 分数基线。

输出 data/human/style_baseline.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUM = os.path.join(ROOT, "data", "human")
SOURCES = ["tech_blog.jsonl", "human_news.jsonl", "human_social.jsonl"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stats(rows, kind="style+logic"):
    X = FEAT.extract_many([r["text"] for r in rows], kind)
    return X.mean(0), X.std(0) + 1e-6, len(rows)


def main():
    out = {"feature_kind": "style+logic", "names": FEAT.names("style+logic"), "baselines": {}}
    allX = []
    for fn in SOURCES:
        p = os.path.join(HUM, fn)
        if not os.path.exists(p):
            continue
        rows = load(p)
        X = FEAT.extract_many([r["text"] for r in rows], "style+logic")
        allX.append(X)
        mu, sd, n = stats(rows)
        out["baselines"][fn.replace(".jsonl", "")] = {"mean": mu.tolist(), "std": sd.tolist(), "n": n}
        print(fn, n)
    X = np.vstack(allX)
    out["baselines"]["all"] = {"mean": X.mean(0).tolist(), "std": (X.std(0) + 1e-6).tolist(), "n": len(X)}
    # 重点话语特征（用于报告聚合）
    out["focus"] = ["reveal", "contrast", "eval", "conclusion_wrap"]
    json.dump(out, open(os.path.join(HUM, "style_baseline.json"), "w"), ensure_ascii=False)
    print("saved -> data/human/style_baseline.json  (all n=%d)" % len(X))


if __name__ == "__main__":
    main()
