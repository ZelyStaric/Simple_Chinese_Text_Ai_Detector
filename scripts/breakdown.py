#!/usr/bin/env python3
"""把留出生成器(Qwen)样本按域拆开看检出率，并与人类留出算整体 AUROC。"""
import os, sys, json, argparse, numpy as np, torch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FEAT
from train_hybrid import Hybrid, ac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def predict(model, tok, rows, dev, meta):
    ids = [tok(r["text"], truncation=True, max_length=meta["max_len"])["input_ids"] for r in rows]
    st = FEAT.extract_many([r["text"] for r in rows], meta.get("feature_kind", "style"))
    if meta["no_style"]:
        st = np.zeros((len(st), 0), dtype=np.float32)
    else:
        st = ((st - np.array(meta["style_mean"])) / np.array(meta["style_std"])).astype(np.float32)
    ps = []
    with torch.no_grad(), ac(dev):
        for i in range(0, len(rows), 64):
            seqs = [s[:meta["max_len"]] for s in ids[i:i + 64]]
            L = max(len(s) for s in seqs)
            inp = torch.zeros(len(seqs), L, dtype=torch.long); am = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            lg = model(inp.to(dev), am.to(dev), torch.tensor(st[i:i + 64]).to(dev))
            ps.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "processed_v2"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.model, "meta.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Hybrid(meta["base"], 0 if meta["no_style"] else len(meta["style_mean"])).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev)); model.eval()

    rows = load(os.path.join(a.data_dir, "test_unseen_gen.jsonl"))
    p = predict(model, tok, rows, dev, meta)
    print(f"== {os.path.basename(a.model)} 留出生成器(Qwen) 分域检出率 ==")
    doms = {}
    for r, pi in zip(rows, p):
        doms.setdefault(r["domain"], []).append(pi)
    for d, ps in sorted(doms.items(), key=lambda x: -len(x[1])):
        ps = np.array(ps)
        print(f"  {d:18s} n={len(ps):4d} 检出率={(ps>=.5).mean():.4f} mean_p={ps.mean():.3f}")

    humans = [r["text"] for r in load(os.path.join(a.data_dir, "dev.jsonl")) if r["label"] == 0]
    ph = predict(model, tok, [{"text": t} for t in humans], dev, meta)
    y = np.concatenate([np.ones(len(p)), np.zeros(len(ph))])
    pa = np.concatenate([p, ph])
    print(f"  --- 合并 AUROC(跨生成器) = {roc_auc_score(y, pa):.4f}  人类 FPR = {(ph>=.5).mean():.4f}")


if __name__ == "__main__":
    main()
