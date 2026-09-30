#!/usr/bin/env python3
"""从 THUCNews(cnews) 与 知乎 构建真人「新闻」「社交」语料。"""
import os, re, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "human", "raw")
OUT = os.path.join(ROOT, "data", "human")
CJK = re.compile(r"[\u4e00-\u9fff]")
random.seed(0)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def build_news(n=4000):
    rows = []
    for line in open(os.path.join(RAW, "cnews.train.txt"), encoding="utf-8"):
        if "\t" not in line:
            continue
        _, text = line.split("\t", 1)
        t = norm(text)
        if 80 <= len(t) <= 4000 and cjk_ratio(t) > 0.3:
            rows.append(dict(text=t, label=0, generator="human", domain="news",
                             source="thucnews"))
    random.shuffle(rows)
    rows = rows[:n]
    out = os.path.join(OUT, "human_news.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("news", len(rows), "->", out)


def build_social(n=4000):
    rows = []
    for line in open(os.path.join(RAW, "zhihu_test.json"), encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = norm(d.get("content", ""))
        if 50 <= len(t) <= 1500 and cjk_ratio(t) > 0.3:
            rows.append(dict(text=t, label=0, generator="human", domain="social",
                             source="zhihu"))
    random.shuffle(rows)
    rows = rows[:n]
    out = os.path.join(OUT, "human_social.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("social", len(rows), "->", out)


if __name__ == "__main__":
    build_news()
    build_social()
