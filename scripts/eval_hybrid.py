#!/usr/bin/env python3
"""评测混合检测器：dev / 留出域 / 留出生成器 / 纯AI域检出率。"""
import os, sys, json, argparse, numpy as np, torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT
from train_hybrid import Hybrid, ac
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def predict(model, tok, rows, dev, max_len):
    ids = [tok(r["text"], truncation=True, max_length=max_len)["input_ids"] for r in rows]
    st = FEAT.extract_many([r["text"] for r in rows], model_meta.get("feature_kind", "style"))
    mu = np.array(model_meta["style_mean"]); sd = np.array(model_meta["style_std"])
    st = np.zeros((len(st), 0), dtype=np.float32) if model_meta["no_style"] else ((st - mu) / sd).astype(np.float32)
    ps = []
    with torch.no_grad(), ac(dev):
        for i in range(0, len(rows), 64):
            seqs = [s[:max_len] for s in ids[i:i + 64]]
            L = max(len(s) for s in seqs)
            inp = torch.zeros(len(seqs), L, dtype=torch.long); am = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            stb = torch.tensor(st[i:i + 64]).to(dev)
            lg = model(inp.to(dev), am.to(dev), stb)
            ps.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def report(name, p, y):
    pred = (p >= .5).astype(int)
    try:
        au = roc_auc_score(y, p)
    except ValueError:
        au = float("nan")
    print(f"{name:34s} n={len(y):5d} acc={accuracy_score(y,pred):.4f} "
          f"f1={f1_score(y,pred,average='macro'):.4f} auroc={au:.4f}")


def main():
    global model_meta
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed_v2"))
    ap.add_argument("--gen_dir", default=os.path.join(ROOT, "data", "generated"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_meta = json.load(open(os.path.join(a.model, "meta.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Hybrid(model_meta["base"], 0 if model_meta["no_style"] else len(model_meta["style_mean"])).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    model.eval()

    for f in ["dev.jsonl", "test_unseen_dom.jsonl", "test_unseen_gen.jsonl"]:
        p = os.path.join(a.data_dir, f)
        if os.path.exists(p):
            rows = load(p); pv = predict(model, tok, rows, dev, model_meta["max_len"])
            report(f, pv, np.array([r["label"] for r in rows]))

    print("\n-- 纯 AI 域检出率（无人类配对）--")
    for f in sorted(os.listdir(a.data_dir)):
        if f.startswith("evalonly_"):
            rows = load(os.path.join(a.data_dir, f)); pv = predict(model, tok, rows, dev, model_meta["max_len"])
            print(f"{f:34s} n={len(rows):5d} 检出率={(pv>=.5).mean():.4f} mean_p={pv.mean():.3f}")


if __name__ == "__main__":
    main()
