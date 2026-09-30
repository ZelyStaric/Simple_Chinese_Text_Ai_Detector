#!/usr/bin/env python3
"""v3 数据集：在 v2 基础上新增 tech_blog 域（真人技术博客 vs 同题 AI 技术博客）。

- 人类锚点：data/human/tech_blog.jsonl（阮一峰/云风/张鑫旭/Draven，<=2022）
- AI（训练生成器）：data/generated/internlm_techblog.jsonl
- AI（留出生成器）：data/generated/qwen_techblog.jsonl
留出生成器 = qwen3.8-27b；留出域 = hc3_psychology；纯 AI 域(news/social) 仅评测。
"""
import json, re, os, hashlib, random
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
GEN = os.path.join(ROOT, "data", "generated")
HUM = os.path.join(ROOT, "data", "human")
OUT = os.path.join(ROOT, "data", "processed_v3")
os.makedirs(OUT, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
MIN_LEN, MAX_LEN = 50, 15000
random.seed(42)

UNSEEN_GENERATORS = {"qwen3.8-27b"}
UNSEEN_DOMAIN = "hc3_psychology"
AI_ONLY_DOMAINS = {"news_gen", "social_gen"}


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


def load_generated():
    files = {
        "wiki_articles.jsonl": ("human", "wiki_article"),
        "qwen_articles.jsonl": ("qwen3.8-27b", "wiki_article"),
        "internlm_articles.jsonl": ("internlm2.5-7b", "wiki_article"),
        "qwen_qa.jsonl": ("qwen3.8-27b", None),
        "internlm_qa.jsonl": ("internlm2.5-7b", None),
        "qwen_news.jsonl": ("qwen3.8-27b", "news_gen"),
        "internlm_news.jsonl": ("internlm2.5-7b", "news_gen"),
        "qwen_social.jsonl": ("qwen3.8-27b", "social_gen"),
        "internlm_social.jsonl": ("internlm2.5-7b", "social_gen"),
        "internlm_techblog.jsonl": ("internlm2.5-7b", "tech_blog"),
        "qwen_techblog.jsonl": ("qwen3.8-27b", "tech_blog"),
    }
    recs = []
    for fn, (gen, dom) in files.items():
        for r in read_jsonl(os.path.join(GEN, fn)):
            t = norm(r.get("text", ""))
            if not t:
                continue
            d = dom or r.get("domain", "qa_gen")
            recs.append(dict(text=t, label=int(r.get("label", 1)),
                             generator=r.get("generator", gen), domain=d,
                             source=r.get("source", "local-gen")))
    return recs


def load_human_techblog():
    recs = []
    for r in read_jsonl(os.path.join(HUM, "tech_blog.jsonl")):
        t = norm(r.get("text", ""))
        if not t:
            continue
        recs.append(dict(text=t, label=0, generator="human", domain="tech_blog",
                         source="human_blog"))
    return recs


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not rows:
        print(f"{os.path.basename(path):28s} EMPTY"); return
    print(f"{os.path.basename(path):28s} n={len(rows):6d}  label={dict(Counter(x['label'] for x in rows))}")
    print(f"{'':28s} gen={dict(Counter(x['generator'] for x in rows))}")
    print(f"{'':28s} dom={dict(Counter(x['domain'] for x in rows))}")


def main():
    recs = dedup(load_hc3() + load_semeval() + load_human_techblog() + load_generated())
    print(f"合并去重后总数: {len(recs)}\n")

    unseen_gen = [r for r in recs if r["generator"] in UNSEEN_GENERATORS]
    unseen_dom = [r for r in recs if r["domain"] == UNSEEN_DOMAIN
                  and r["generator"] not in UNSEEN_GENERATORS]
    trainable = [r for r in recs
                 if r["generator"] not in UNSEEN_GENERATORS
                 and r["domain"] != UNSEEN_DOMAIN
                 and r["domain"] not in AI_ONLY_DOMAINS]
    random.shuffle(trainable)
    n_dev = int(len(trainable) * 0.1)
    dev, train = trainable[:n_dev], trainable[n_dev:]

    dump(os.path.join(OUT, "train.jsonl"), train)
    dump(os.path.join(OUT, "dev.jsonl"), dev)
    dump(os.path.join(OUT, "test_unseen_gen.jsonl"), unseen_gen)
    dump(os.path.join(OUT, "test_unseen_dom.jsonl"), unseen_dom)
    for dom in ["news_gen", "social_gen"]:
        for gen in ["qwen3.8-27b", "internlm2.5-7b"]:
            rows = [r for r in recs if r["domain"] == dom and r["generator"] == gen]
            tag = gen.split("-")[0]
            dump(os.path.join(OUT, f"evalonly_{tag}_{dom}.jsonl"), rows)


if __name__ == "__main__":
    main()
