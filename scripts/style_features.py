#!/usr/bin/env python3
"""AI 味「表层风格特征」提取 —— 直击用户关注点。

覆盖：语气/套话、符号用法（破折号/冒号/分号）、括号内名词解释、表格、
比喻、排比、句式节奏、第一人称、markdown 结构。
返回固定顺序的 numpy 向量 + 名称表。
"""
import re
import numpy as np

CJK = r"\u4e00-\u9fff"
FW = r"\u3000-\u303f\uff00-\uffef"  # 全角标点区

# 套话/口号（来自 blog-writing-style.md 及常见 AI 腔）
AI_PHRASES = [
    "不仅", "而且", "值得注意", "值得一提", "综上", "总而言之", "总的来说", "总结一下",
    "与此同时", "换句话说", "在某种程度上", "由此可见", "不难看出", "概括一下",
    "一句话概括", "简单来说", "结论是", "原因很简单", "后果是",
    "赋能", "闭环", "抓手", "范式", "生态", "利器", "助力", "打磨", "落地",
    "作为一个", "随着", "在当今", "让我们一起", "首先", "其次", "再者", "最后",
    "强大", "优雅", "极致", "无缝", "完美", "丝滑", "高效便捷", "不可或缺",
    "至关重要", "毋庸置疑", "举足轻重", "深远的影响", "质的飞跃", "显著提升",
    "需要注意的是", "需要指出的是", "换句话说", "从某种程度上", "无疑",
]
# 比喻/类比标记
SIMILE = ["就像", "如同", "好比", "仿佛", "宛如", "恰似", "犹如", "像是", "类似于"]

# 名词（解释）型括号：紧邻中文/英文词，括号内 2~40 字且不含句号
GLOSS_RE = re.compile(rf"[{CJK}A-Za-z0-9][（(]([^（()）]{{2,40}})[)）]")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.、)])\s+")
HEAD_RE = re.compile(r"^\s*#{1,6}\s+")
# 排比：连续 ≥3 个以相同 2 字前缀开头的分句（简易检测）

FEATURE_NAMES = [
    "len_log", "em_dash", "colon_fw", "semicolon", "ellipsis",
    "paren_fw", "paren_ascii", "gloss", "quotes", "bold",
    "list_rate", "header_rate", "table_row", "table_sep",
    "code_fence", "ai_phrase", "simile", "parallel",
    "de_ratio", "i_ratio", "we_ratio",
    "sent_len_mean", "sent_len_cv", "short_sent_ratio",
    "para_len_cv", "newline_rate", "exclam_rate", "question_rate",
]


def _rate(cnt, n_chars):
    return 1000.0 * cnt / max(1, n_chars)


def extract(text):
    t = text
    n = len(t)
    lines = t.split("\n")

    em_dash = len(re.findall(r"——|—", t))
    colon_fw = len(re.findall(r"：", t))
    semicolon = len(re.findall(r"；", t))
    ellipsis = len(re.findall(r"……|\.\.\.", t))
    paren_fw = len(re.findall(r"[（(]", t))
    paren_ascii = len(re.findall(r"\(", t))
    gloss = len(GLOSS_RE.findall(t))
    quotes = len(re.findall(r"[“”「」『』\"']", t))
    bold = len(re.findall(r"\*\*", t)) // 2
    list_cnt = sum(1 for l in lines if LIST_RE.match(l))
    head_cnt = sum(1 for l in lines if HEAD_RE.match(l))
    table_row = sum(1 for l in lines if "|" in l)
    table_sep = sum(1 for l in lines if TABLE_SEP_RE.match(l))
    code_fence = t.count("```") // 2
    ai_phrase = sum(t.count(p) for p in AI_PHRASES)
    simile = sum(t.count(p) for p in SIMILE)
    # 排比：以相同 2 字开头、以「，」分隔的连续短句 >=3
    parallel = 0
    for m in re.finditer(r"(?:[\u4e00-\u9fff]{2})[^，。！？\n]{0,20}[，。]", t):
        pass
    segs = re.split(r"[，。！？；\n]", t)
    prefixes = {}
    for s in segs:
        s = s.strip()
        if 2 <= len(s) <= 25:
            p = s[:2]
            prefixes[p] = prefixes.get(p, 0) + 1
    parallel = sum(v - 1 for v in prefixes.values() if v >= 3)

    de_ratio = t.count("的") / max(1, n)
    i_ratio = (t.count("我") - t.count("我们")) / max(1, n)
    we_ratio = t.count("我们") / max(1, n)

    sents = [s for s in re.split(r"[。！？\n]", t) if s.strip()]
    slens = [len(s) for s in sents] or [0]
    slen_mean = float(np.mean(slens))
    slen_cv = float(np.std(slens) / (slen_mean + 1e-6))
    short_ratio = float(np.mean([1 if x < 15 else 0 for x in slens]))

    paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    plens = [len(p) for p in paras] or [0]
    para_cv = float(np.std(plens) / (np.mean(plens) + 1e-6))

    return np.array([
        np.log1p(n), _rate(em_dash, n), _rate(colon_fw, n), _rate(semicolon, n),
        _rate(ellipsis, n), _rate(paren_fw, n), _rate(paren_ascii, n), _rate(gloss, n),
        _rate(quotes, n), _rate(bold, n), _rate(list_cnt, n), _rate(head_cnt, n),
        _rate(table_row, n), _rate(table_sep, n), _rate(code_fence, n),
        _rate(ai_phrase, n), _rate(simile, n), _rate(parallel, n),
        de_ratio, i_ratio, we_ratio,
        slen_mean, slen_cv, short_ratio, para_cv,
        t.count("\n") / max(1, n), _rate(t.count("！"), n), _rate(t.count("？"), n),
    ], dtype=np.float32)


def extract_many(texts):
    return np.stack([extract(x) for x in texts])


if __name__ == "__main__":
    import sys, json
    doc = sys.stdin.read()
    f = extract(doc)
    for name, val in zip(FEATURE_NAMES, f):
        print(f"{name:16s} {val:.4f}")
