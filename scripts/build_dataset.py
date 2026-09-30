#!/usr/bin/env python3
"""把 HC3-Chinese + SemEval-2024-zh 汇总成统一格式，并按生成器/领域切分。

输出 data/processed/*.jsonl，字段：text,label(1=AI,0=人),generator,domain,source
"""
import json, re, os, hashlib, random
from collections import Counter
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
MIN_LEN, MAX_LEN = 50, 10000
random.seed(42)

# 留出的生成器（完全不进训练），用来测跨生成器泛化
UNSEEN_GENERATORS = {"bloomz", "dolly", "cohere"}
# 留出的领域（完全不进训练），用来测跨领域泛化
UNSEEN_DOMAIN = "hc3_psychology"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def get(recs):
    recs = [r for r in recs if MIN_LEN <= len(r["text"]) <= MAX_LEN]
    seen, uniq = set(), []
    for r in recs:
        h = hashlib.sha1(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(r)
    return uniq


def load_hc3():
    recs = []
    for dom in ["baike", "finance", "law", "medicine", "psychology", "open_qa", "nlpcc_dbqa"]:
        p = os.path.join(RAW, "hc3", f"{dom}.jsonl")
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            for a in d.get("human_answers") or []:
                recs.append(dict(text=norm(a), label=0, generator="human",
                                 domain=f"hc3_{dom}", source="hc3"))
            for a in d.get("chatgpt_answers") or []:
                recs.append(dict(text=norm(a), label=1, generator="chatgpt",
                                 domain=f"hc3_{dom}", source="hc3"))
    return recs


def load_semeval():
    recs = []
    df = pd.read_parquet(os.path.join(RAW, "semeval", "subtaskA_multilingual_train.parquet"))
    for _, r in df.iterrows():
        t = norm(r["text"])
        if cjk_ratio(t) < 0.10:  # 只留中文文本
            continue
        gen = "human" if r["model"] == "human" else str(r["model"]).lower()
        recs.append(dict(text=t, label=int(r["label"]), generator=gen,
                         domain="semeval_zh", source="semeval"))
    return recs


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(x["generator"] for x in rows)
    d = Counter(x["domain"] for x in rows)
    l = Counter(x["label"] for x in rows)
    print(f"{os.path.basename(path):28s} n={len(rows):7d}  label={dict(l)}")
    print(f"{'':28s} gen={dict(c)}")
    print(f"{'':28s} dom={dict(d)}")


def main():
    recs = get(load_hc3() + load_semeval())
    print(f"合并去重后总数: {len(recs)}\n")

    unseen_gen = [r for r in recs if r["generator"] in UNSEEN_GENERATORS]
    unseen_dom = [r for r in recs if r["domain"] == UNSEEN_DOMAIN
                  and r["generator"] not in UNSEEN_GENERATORS]
    rest = [r for r in recs if r["generator"] not in UNSEEN_GENERATORS
            and r["domain"] != UNSEEN_DOMAIN]

    random.shuffle(rest)
    n_dev = int(len(rest) * 0.1)
    dev, train = rest[:n_dev], rest[n_dev:]

    dump(os.path.join(OUT, "train.jsonl"), train)
    dump(os.path.join(OUT, "dev.jsonl"), dev)
    dump(os.path.join(OUT, "test_unseen_gen.jsonl"), unseen_gen)
    dump(os.path.join(OUT, "test_unseen_dom.jsonl"), unseen_dom)


if __name__ == "__main__":
    main()
