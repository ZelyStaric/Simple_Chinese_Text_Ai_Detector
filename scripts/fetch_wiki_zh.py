#!/usr/bin/env python3
"""抓中文维基百科条目首段（人类长文），作为长文域的「人写」样本。"""
import os, json, argparse, re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJK = re.compile(r"[\u4e00-\u9fff]")


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def strip_wiki(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"==+[^=]+==+", "", s)
    s = re.sub(r"'''?", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "generated", "wiki_articles.jsonl"))
    ap.add_argument("--min_len", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=1500)
    a = ap.parse_args()

    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f = open(a.out, "w", encoding="utf-8")
    n = 0
    for ex in ds:
        title = ex.get("title", "").strip()
        text = strip_wiki(ex.get("text", ""))
        if not title or not text:
            continue
        # 首个自然段
        para = text.split("\n")[0].strip()
        if not (a.min_len <= len(para) <= a.max_len):
            continue
        if cjk_ratio(para) < 0.5:
            continue
        f.write(json.dumps(dict(text=para, title=title, label=0,
                                generator="human", domain="wiki_article",
                                source="wikipedia"), ensure_ascii=False) + "\n")
        n += 1
        if n % 200 == 0:
            print(f"{n}", flush=True)
        if n >= a.n:
            break
    f.close()
    print(f"wrote {n} -> {a.out}")


if __name__ == "__main__":
    main()
