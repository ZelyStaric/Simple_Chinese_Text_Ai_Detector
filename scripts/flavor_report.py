#!/usr/bin/env python3
"""AI 味报告：分类器概率 ⋈ 话语脚手架 z 分数（可融合），并逐段定位。

不依赖分类器学会话语脚手架——把它作为独立可解释信号，按 noisy-or 融合。
用法:
  ./env/bin/python scripts/flavor_report.py --text data/blog/zelystaric_rdna3.txt \
      --model models/hybrid3_v5 --baseline tech_blog --lam 1.0
"""
import os, sys, re, json, argparse, numpy as np, torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT
import logic_features as L
from train_hybrid import Hybrid, ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUM = os.path.join(ROOT, "data", "human")
FOCUS = ["reveal", "contrast", "eval", "conclusion_wrap"]
TRIGGERS = {
    "报幕/揭晓": L.REVEAL,
    "对比公式": [p for p in L.CONTRAST if not any(c in p for c in ".*?()[]")],
    "评价收口": L.EVAL,
    "显式因果": L.CAUSAL,
}
WRAP = ["真正", "其余", "总的来说", "综上", "总而言之", "总结", "以上", "这轮", "归根", "最后"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def strip_structure(t):
    """文档体裁模式：剥离 markdown/结构标记，只留文字，避免被格式特征带偏。"""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"^\s*\|.*$", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+[.)、]\s+", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def predict_windows(model, tok, text, meta, dev):
    ids = tok(text, truncation=False)["input_ids"]
    W = meta["max_len"]; S = max(64, W // 2)
    wins = [ids[i:i + W] for i in range(0, max(1, len(ids)), S) if ids[i:i + W]]
    if meta.get("no_style"):
        st = np.zeros(0, dtype=np.float32)
    else:
        st = ((FEAT.extract(text, meta["feature_kind"]) - np.array(meta["style_mean"])) /
              np.array(meta["style_std"])).astype(np.float32)
    P = []
    with torch.no_grad(), ac(dev):
        for w in wins:
            stb = torch.zeros(1, 0).to(dev) if st.shape[0] == 0 else torch.tensor(st)[None].to(dev)
            lg = model(torch.tensor([w]).to(dev), torch.ones(1, len(w), dtype=torch.long).to(dev), stb)
            P.append(torch.softmax(lg.float(), -1)[0].cpu().numpy())
    return np.mean(P, axis=0), len(wins)


def locate(text):
    rows = []
    for gi, para in enumerate([p for p in text.split("\n") if p.strip()]):
        hits = []
        for name, pats in TRIGGERS.items():
            for p in pats:
                if p in para:
                    hits.append((name, p))
        if gi and any(w in para for w in WRAP):   # 收尾段
            hits.append(("总分总收尾", next(w for w in WRAP if w in para)))
        if hits:
            rows.append((gi, para.strip()[:60], hits))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "encoder_v9"))
    ap.add_argument("--baseline", default="tech_blog", choices=["tech_blog", "all", "human_news", "human_social"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mode", choices=["prose", "doc"], default="prose",
                    help="doc：剥离 markdown/结构标记，适合 README/文档/PPT（避免格式特征误伤）")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    text = open(a.text, encoding="utf-8").read()
    text_n = re.sub(r"\s+", " ", text).strip()          # 与训练一致的空白归一化
    text_m = strip_structure(text_n) if a.mode == "doc" else text_n
    base = json.load(open(os.path.join(HUM, "style_baseline.json")))
    b = base["baselines"][a.baseline]
    mu, sd = np.array(b["mean"]), np.array(b["std"])
    names = base["names"]

    meta = json.load(open(os.path.join(a.model, "meta.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    nlab = meta.get("num_labels", 2)
    model = Hybrid(meta["base"], len(meta["style_mean"]), num_labels=nlab).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev)); model.eval()
    P, nwin = predict_windows(model, tok, text_m, meta, dev)
    clf_ai = float(1 - P[0]) if nlab == 2 else float(1 - P[0])   # 非人类即 AI 味

    x = FEAT.extract(text_m, base["feature_kind"])
    z = (x - mu) / sd
    zc = {f: float(z[names.index(f)]) for f in FOCUS}
    agg = float(np.mean([max(0.0, v) for v in zc.values()]))
    scaffold_prob = float(sigmoid((agg - 2.0) / 0.8))
    final = 1 - (1 - clf_ai) * (1 - a.lam * scaffold_prob)

    print(f"===== AI 味报告：{os.path.basename(a.text)} =====")
    print(f"分类器（{nlab} 类，{nwin} 窗口）：", end="")
    if nlab == 3:
        print(f"human={P[0]:.3f}  AI={P[1]:.3f}  AI-edited={P[2]:.3f}")
    else:
        print(f"human={P[0]:.3f}  AI={P[1]:.3f}")
    print(f"话语脚手架 z（基线={a.baseline}, n={b['n']}）：")
    for f in FOCUS:
        flag = "  ⚠" if zc[f] > 3 else ""
        print(f"    {f:16s} x={x[names.index(f)]:7.3f}  mean={mu[names.index(f)]:6.3f}  z={zc[f]:+7.2f}{flag}")
    print(f"  聚合 z = {agg:.2f}  ->  scaffold_prob = {scaffold_prob:.3f}")
    print(f"最终 AI 味分 = {final:.3f}   (clf_ai={clf_ai:.3f}, λ={a.lam})")

    loc = locate(text)
    if loc:
        print(f"\n逐段定位（{len(loc)} 段命中）：")
        for gi, snip, hits in loc:
            h = "、".join(f"{n}「{p}」" for n, p in hits[:6])
            print(f"  [段{gi}] {h}    ← {snip}...")


if __name__ == "__main__":
    main()
