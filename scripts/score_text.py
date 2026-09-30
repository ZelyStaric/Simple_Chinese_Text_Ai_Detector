#!/usr/bin/env python3
"""给一整篇文章打「AI 分」+ 风格拆解（对照维基人类 / AI 科普文均值）。

用法:
  ./env/bin/python scripts/score_text.py --model models/hybrid_v2 --text data/blog/zelystaric_rdna3.txt
"""
import os, sys, json, re, argparse, numpy as np, torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_features import extract, extract_many, FEATURE_NAMES
import features as FEAT
from train_hybrid import Hybrid, ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH = ["ai_phrase", "parallel", "table_row", "bold", "header_rate", "list_rate",
         "sent_len_cv", "short_sent_ratio", "sent_len_mean", "paren_fw", "gloss",
         "em_dash", "colon_fw", "de_ratio"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def predict_windows(model, tok, text, meta, dev):
    ids = tok(text, truncation=False)["input_ids"]
    W = meta["max_len"]; S = max(64, W // 2)
    wins = [ids[i:i + W] for i in range(0, max(1, len(ids)), S)]
    wins = [w for w in wins if w]
    st_row = FEAT.extract(text, meta.get("feature_kind", "style"))
    mu = np.array(meta["style_mean"]); sd = np.array(meta["style_std"])
    ps = []
    with torch.no_grad(), ac(dev):
        for w in wins:
            inp = torch.tensor([w]).to(dev); am = torch.ones_like(inp)
            if meta["no_style"]:
                st = torch.zeros(1, 0).to(dev)
            else:
                st = torch.tensor(((st_row - mu) / sd).astype(np.float32))[None].to(dev)
            ps.append(torch.softmax(model(inp, am, st).float(), -1)[0, 1].item())
    return np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--gen_dir", default=os.path.join(ROOT, "data", "generated"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.model, "meta.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Hybrid(meta["base"], 0 if meta["no_style"] else len(meta["style_mean"])).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev)); model.eval()

    text = open(a.text, encoding="utf-8").read()
    text_n = re.sub(r"\s+", " ", text).strip()
    ps = predict_windows(model, tok, text_n, meta, dev)
    print(f"== {os.path.basename(a.model)}  «{os.path.basename(a.text)}» ==")
    print(f"  窗口 n={len(ps)}  P(AI) mean={ps.mean():.3f} max={ps.max():.3f} "
          f"≥0.5 比例={(ps>=.5).mean():.2f}")

    Xh = extract_many([r["text"] for r in load(os.path.join(a.gen_dir, "wiki_articles.jsonl"))])
    Xa = extract_many([r["text"] for r in load(os.path.join(a.gen_dir, "qwen_articles.jsonl"))]
                      + [r["text"] for r in load(os.path.join(a.gen_dir, "internlm_articles.jsonl"))])
    x = extract(text_n)
    print(f"\n  {'feature':16s} {'本篇':>8s} {'维基人':>8s} {'AI科普':>8s}")
    for n in WATCH:
        i = FEATURE_NAMES.index(n)
        print(f"  {n:16s} {x[i]:8.3f} {Xh[:,i].mean():8.3f} {Xa[:,i].mean():8.3f}")


if __name__ == "__main__":
    main()
