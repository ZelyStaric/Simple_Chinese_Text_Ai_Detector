#!/usr/bin/env python3
"""v6 数据集（二分类 human/AI）：加入「话语风格」AI 样本，博客排除出训练（作公平测试）。

- AI 样本 = 普通 AI + AI-edited + 话语风格(flavor) 全部并为一类。
- 留出：Qwen 生成/改写/风格样本；留出域 psychology；用户博客(单独文件，不进训练)。
"""
import json, re, os, hashlib, random
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
GEN = os.path.join(ROOT, "data", "generated")
HUM = os.path.join(ROOT, "data", "human")
OUT = os.path.join(ROOT, "data", "processed_v8")
os.makedirs(OUT, exist_ok=True)

CJK = re.compile(r"[\u4e00-\u9fff]")
MIN_LEN, MAX_LEN = 50, 15000
random.seed(42)
UNSEEN = {"qwen3.8-27b", "qwen3.8-27b-edited", "qwen3.8-27b-flavor"}
UNSEEN_DOMAIN = "hc3_psychology"
BLOG_GEN = "zelystaric-indie"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def read(p):
    if not os.path.exists(p):
        print(f"  (缺 {os.path.basename(p)})")
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_all():
    recs = []
    for dom in ["baike", "finance", "law", "medicine", "psychology", "open_qa", "nlpcc_dbqa"]:
        for line in open(os.path.join(RAW, "hc3", f"{dom}.jsonl"), encoding="utf-8"):
            d = json.loads(line)
            for x in d.get("human_answers") or []:
                recs.append(dict(text=norm(x), label=0, generator="human", domain=f"hc3_{dom}"))
            for x in d.get("chatgpt_answers") or []:
                recs.append(dict(text=norm(x), label=1, generator="chatgpt", domain=f"hc3_{dom}"))
    df = pd.read_parquet(os.path.join(RAW, "semeval", "subtaskA_multilingual_train.parquet"))
    for _, r in df.iterrows():
        t = norm(r["text"])
        if cjk_ratio(t) < 0.10:
            continue
        gen = "human" if r["model"] == "human" else str(r["model"]).lower()
        recs.append(dict(text=t, label=int(r["label"]), generator=gen, domain="semeval_zh"))
    for fn, dom in [("tech_blog.jsonl", "tech_blog"), ("human_news.jsonl", "news"), ("human_social.jsonl", "social")]:
        for r in read(os.path.join(HUM, fn)):
            recs.append(dict(text=norm(r["text"]), label=0, generator="human", domain=dom))

    gen_spec = [
        ("qwen_articles.jsonl", "qwen3.8-27b", "wiki_article"),
        ("internlm_articles.jsonl", "internlm2.5-7b", "wiki_article"),
        ("qwen_qa.jsonl", "qwen3.8-27b", None), ("internlm_qa.jsonl", "internlm2.5-7b", None),
        ("qwen_news.jsonl", "qwen3.8-27b", "news"), ("internlm_news.jsonl", "internlm2.5-7b", "news"),
        ("qwen_social.jsonl", "qwen3.8-27b", "social"), ("internlm_social.jsonl", "internlm2.5-7b", "social"),
        ("internlm_techblog.jsonl", "internlm2.5-7b", "tech_blog"),
        ("qwen_techblog.jsonl", "qwen3.8-27b", "tech_blog"),
        ("internlm_edited_techblog.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("ed_med_internlm.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("ed_light.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("ed_heavy.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("ed_med_human.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("ed_med_qwen.jsonl", "internlm2.5-7b-edited", "tech_blog"),
        ("internlm_flavor.jsonl", "internlm2.5-7b-flavor", "tech_blog"),
        ("gpt4_zh.jsonl", "gpt4", "gpt4_writing"),
        ("self_article.jsonl", "deepseek-v4.1-flash", "tech_blog"),
        ("self_edited.jsonl", "deepseek-v4.1-flash-edited", "tech_blog"),
        ("qwen_edited_techblog.jsonl", "qwen3.8-27b-edited", "tech_blog"),
        ("qwen_flavor.jsonl", "qwen3.8-27b-flavor", "tech_blog"),
        ("blog_edited.jsonl", BLOG_GEN, "tech_blog"),
    ]
    for fn, gen, dom in gen_spec:
        for r in read(os.path.join(GEN, fn)):
            t = norm(r.get("text", ""))
            if t:
                recs.append(dict(text=t, label=int(r.get("label", 1)), generator=gen,
                                 domain=(dom or r.get("domain", "qa_gen"))))
    return recs


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


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not rows:
        print(f"{os.path.basename(path):26s} EMPTY"); return
    print(f"{os.path.basename(path):26s} n={len(rows):6d} lab={dict(Counter(x['label'] for x in rows))} dom={dict(Counter(x['domain'] for x in rows))}")


def main():
    recs = dedup(load_all())
    for r in recs:                       # 合并为二分类
        r["label"] = 0 if r["label"] == 0 else 1
    blog = [r for r in recs if r["generator"] == BLOG_GEN]
    recs = [r for r in recs if r["generator"] != BLOG_GEN]
    cap = {"news": 800, "social": 800}
    capped, cnt = [], Counter()
    for r in recs:
        if r["label"] == 0 and r["domain"] in cap:
            if cnt[r["domain"]] >= cap[r["domain"]]:
                continue
            cnt[r["domain"]] += 1
        capped.append(r)
    recs = capped
    print("总数", len(recs), "| blog(测试)", len(blog), "\n")

    unseen = [r for r in recs if r["generator"] in UNSEEN]
    unseen_dom = [r for r in recs if r["domain"] == UNSEEN_DOMAIN and r["generator"] not in UNSEEN]
    trainable = [r for r in recs if r["generator"] not in UNSEEN and r["domain"] != UNSEEN_DOMAIN]
    random.shuffle(trainable)
    n_dev = int(len(trainable) * 0.1)
    dev, train = trainable[:n_dev], trainable[n_dev:]
    dump(os.path.join(OUT, "train.jsonl"), train)
    dump(os.path.join(OUT, "dev.jsonl"), dev)
    dump(os.path.join(OUT, "test_unseen_gen.jsonl"), unseen)
    dump(os.path.join(OUT, "test_unseen_dom.jsonl"), unseen_dom)
    dump(os.path.join(OUT, "test_blog.jsonl"), blog)


if __name__ == "__main__":
    main()
