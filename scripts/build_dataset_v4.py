#!/usr/bin/env python3
"""v4 数据集（三类）：human(0) / AI(1) / AI-edited(2)。

新增：
- 真人新闻（THUCNews 4000）、真人社交（知乎 4000）
- AI 新闻/社交（InternLM 800/800；Qwen 200/200 留出）
- AI 改写技术博客（InternLM 300 训练；Qwen 留出）
留出生成器 = qwen3.8-27b；留出域 = hc3_psychology。
"""
import json, re, os, hashlib, random
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
GEN = os.path.join(ROOT, "data", "generated")
HUM = os.path.join(ROOT, "data", "human")
OUT = os.path.join(ROOT, "data", "processed_v4")
os.makedirs(OUT, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
MIN_LEN, MAX_LEN = 50, 15000
random.seed(42)
UNSEEN_GENERATORS = {"qwen3.8-27b", "qwen3.8-27b-edited"}
UNSEEN_DOMAIN = "hc3_psychology"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def dedup(recs):
    seen, out = set(), []
    for r in recs:
        if not (MIN_LEN <= len(r["text"]) <= MAX_LEN):
            continue
        h = hashlib.sha1(r["text"].encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def read_jsonl(p):
    if not os.path.exists(p):
        print(f"  (缺 {os.path.basename(p)})")
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_hc3():
    recs = []
    for dom in ["baike", "finance", "law", "medicine", "psychology", "open_qa", "nlpcc_dbqa"]:
        for line in open(os.path.join(RAW, "hc3", f"{dom}.jsonl"), encoding="utf-8"):
            d = json.loads(line)
            for x in d.get("human_answers") or []:
                recs.append(dict(text=norm(x), label=0, generator="human",
                                 domain=f"hc3_{dom}", source="hc3"))
            for x in d.get("chatgpt_answers") or []:
                recs.append(dict(text=norm(x), label=1, generator="chatgpt",
                                 domain=f"hc3_{dom}", source="hc3"))
    return recs


def load_semeval():
    recs = []
    df = pd.read_parquet(os.path.join(RAW, "semeval", "subtaskA_multilingual_train.parquet"))
    for _, r in df.iterrows():
        t = norm(r["text"])
        if cjk_ratio(t) < 0.10:
            continue
        gen = "human" if r["model"] == "human" else str(r["model"]).lower()
        recs.append(dict(text=t, label=int(r["label"]), generator=gen,
                         domain="semeval_zh", source="semeval"))
    return recs


def load_humans():
    recs = []
    for fn, dom in [("tech_blog.jsonl", "tech_blog"), ("human_news.jsonl", "news"),
                    ("human_social.jsonl", "social")]:
        for r in read_jsonl(os.path.join(HUM, fn)):
            recs.append(dict(text=norm(r["text"]), label=0, generator="human",
                             domain=dom, source=r.get("source", "human")))
    return recs


def load_generated():
    # (file, generator, domain, label)
    spec = [
        ("qwen_articles.jsonl", "qwen3.8-27b", "wiki_article", 1),
        ("internlm_articles.jsonl", "internlm2.5-7b", "wiki_article", 1),
        ("qwen_qa.jsonl", "qwen3.8-27b", None, 1),
        ("internlm_qa.jsonl", "internlm2.5-7b", None, 1),
        ("qwen_news.jsonl", "qwen3.8-27b", "news", 1),
        ("internlm_news.jsonl", "internlm2.5-7b", "news", 1),
        ("qwen_social.jsonl", "qwen3.8-27b", "social", 1),
        ("internlm_social.jsonl", "internlm2.5-7b", "social", 1),
        ("internlm_techblog.jsonl", "internlm2.5-7b", "tech_blog", 1),
        ("qwen_techblog.jsonl", "qwen3.8-27b", "tech_blog", 1),
        ("internlm_edited_techblog.jsonl", "internlm2.5-7b-edited", "tech_blog", 2),
        ("qwen_edited_techblog.jsonl", "qwen3.8-27b-edited", "tech_blog", 2),
    ]
    recs = []
    for fn, gen, dom, lab in spec:
        for r in read_jsonl(os.path.join(GEN, fn)):
            t = norm(r.get("text", ""))
            if not t:
                continue
            d = dom or r.get("domain", "qa_gen")
            recs.append(dict(text=t, label=r.get("label", lab), generator=gen,
                             domain=d, source=r.get("source", "local-gen")))
    return recs


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not rows:
        print(f"{os.path.basename(path):26s} EMPTY"); return
    print(f"{os.path.basename(path):26s} n={len(rows):6d} lab={dict(Counter(x['label'] for x in rows))}")
    print(f"{'':26s} gen={dict(Counter(x['generator'] for x in rows))}")
    print(f"{'':26s} dom={dict(Counter(x['domain'] for x in rows))}")


def main():
    recs = dedup(load_hc3() + load_semeval() + load_humans() + load_generated())
    print(f"合并去重后总数: {len(recs)}\n")

    # 每域人类样本上限，避免与 AI 数量失衡
    human_cap = {"news": 800, "social": 800}
    capped, cnt = [], Counter()
    for r in recs:
        if r["label"] == 0 and r["domain"] in human_cap:
            if cnt[r["domain"]] >= human_cap[r["domain"]]:
                continue
            cnt[r["domain"]] += 1
        capped.append(r)
    recs = capped

    unseen_gen = [r for r in recs if r["generator"] in UNSEEN_GENERATORS]
    unseen_dom = [r for r in recs if r["domain"] == UNSEEN_DOMAIN
                  and r["generator"] not in UNSEEN_GENERATORS]
    trainable = [r for r in recs if r["generator"] not in UNSEEN_GENERATORS
                 and r["domain"] != UNSEEN_DOMAIN]
    random.shuffle(trainable)
    n_dev = int(len(trainable) * 0.1)
    dev, train = trainable[:n_dev], trainable[n_dev:]

    dump(os.path.join(OUT, "train.jsonl"), train)
    dump(os.path.join(OUT, "dev.jsonl"), dev)
    dump(os.path.join(OUT, "test_unseen_gen.jsonl"), unseen_gen)
    dump(os.path.join(OUT, "test_unseen_dom.jsonl"), unseen_dom)


if __name__ == "__main__":
    main()
