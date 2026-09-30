#!/usr/bin/env python3
"""生成 results 可视化图（assets/results.png）。

数据为报告中的最终结果（v9，阈值 0.5）。仅用于文档展示，不依赖训练数据。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "results.png")

# (标签, 百分比)
DETECT = [("AI WeChat-style\n(same-model)", 100.0),
          ("AI tech blog\n(same-model)", 100.0),
          ("Target article\n(held-out)", 100.0)]
FP = [("Human tech blog", 10.0), ("Human WeChat-style", 6.9), ("Human news", 3.0)]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Chinese AI-text detector — final results (v9, threshold 0.5)", fontsize=12)

    x = np.arange(len(DETECT))
    a1.bar(x, [v for _, v in DETECT], color="#2e8b57")
    a1.set_title("AI / AI-edited detection rate  (higher = better)")
    a1.set_xticks(x); a1.set_xticklabels([k for k, _ in DETECT], fontsize=8)
    a1.set_ylim(0, 105); a1.set_ylabel("%")
    for i, (_, v) in enumerate(DETECT):
        a1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    x = np.arange(len(FP))
    a2.bar(x, [v for _, v in FP], color="#c0392b")
    a2.set_title("Human false-positive rate  (lower = better)")
    a2.set_xticks(x); a2.set_xticklabels([k for k, _ in FP], fontsize=8)
    a2.set_ylim(0, 105); a2.set_ylabel("%")
    for i, (_, v) in enumerate(FP):
        a2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=150)
    fig.savefig(os.path.splitext(OUT)[0] + ".svg")
    print("saved", OUT)


if __name__ == "__main__":
    main()
