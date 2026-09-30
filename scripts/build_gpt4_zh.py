#!/usr/bin/env python3
"""从 GPT-4 中文语料（alpaca-gpt4-chinese / Evol-Instruct-Chinese-GPT4）抽取长文，
作为「前沿模型 AI 中文写作」样本（label 1, generator gpt4）。
"""
import os, re, json, hashlib, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "human", "raw")
OUT = os.path.join(ROOT, "data", "generated", "gpt4_zh.jsonl")
CJK = re.compile(r"[\u4e00-\u9fff]")
random.seed(0)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def ok(t):
    if not (300 <= len(t) <= 4000):
        return False
    if cjk_ratio(t) < 0.40:
        return False
    if re.search(r"作为一个\s*(AI|人工智能)|我无法|不能提供|抱歉", t[:40]):
        return False
    if t.count("{") + t.count("```") + t.count("</") > 6:
        return False
    return True


def main():
    rows = []
    # Evol-Instruct: instruction + output
    p = os.path.join(RAW, "evol_gpt4_zh.json")
    if os.path.exists(p):
        d = json.load(open(p))
        random.shuffle(d)
        for x in d:
            t = norm(x.get("output", ""))
            if ok(t):
                rows.append(t)
            if len(rows) >= 4000:
                break
        print("evol kept", len(rows))
    # Alpaca: conversations with gpt turns
    p = os.path.join(RAW, "gpt4_zh_alpaca.json")
    if os.path.exists(p):
        d = json.load(open(p))
        random.shuffle(d)
        n0 = len(rows)
        for x in d:
            for turn in x.get("conversations", []):
                if turn.get("from") == "gpt":
                    t = norm(turn.get("value", ""))
                    if ok(t):
                        rows.append(t); break
            if len(rows) - n0 >= 2000:
                break
        print("alpaca kept", len(rows) - n0)

    seen, uniq = set(), []
    for t in rows:
        h = hashlib.sha1(t.encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h); uniq.append(t)
    random.shuffle(uniq)
    with open(OUT, "w", encoding="utf-8") as f:
        for t in uniq:
            f.write(json.dumps(dict(text=t, label=1, generator="gpt4",
                                    domain="gpt4_writing", source="gpt4-zh"),
                               ensure_ascii=False) + "\n")
    print("wrote", len(uniq), "->", OUT)


if __name__ == "__main__":
    main()
