#!/usr/bin/env python3
"""「语句逻辑 + 语气」特征（与表面符号无关，靠信息结构）。

比 style_features 更贴近用户说的重点：报幕揭晓、显式因果、对比公式、
并列长链、总分总收尾、过度定义、评价收口、句首连词、口气。
"""
import re
import numpy as np

CJK = r"\u4e00-\u9fff"

# 报幕/揭晓语（先把结论抛出来，再展开——AI 高发）
REVEAL = ["原因也清楚", "原因很简单", "后来想清楚", "想清楚了", "结论是", "由此可见",
          "不难看出", "这说明", "也就是说", "换句话说", "归根结底", "说到底", "总结一下",
          "概括一下", "一句话概括", "总的说来", "总的来说", "综上"]
# 显式因果
CAUSAL = ["因此", "所以", "从而", "进而", "由于", "正因为", "原因是", "这就导致", "于是"]
# 对比/对举公式
CONTRAST = ["而不是", "不是", "而是", "本身没有", "本身就", "与其", "不如", "相比之下",
            "与.*?不同", "反而", "而现实", "然而"]
# 过度定义/解释
DEFINE = ["指的是", "是指", "是一种", "所谓", "即为", "也就是说", "即所谓"]
# 评价收口（断言式总结）
EVAL = ["到顶", "差距过大", "并不新奇", "不划算", "性价比低", "不成立", "无效", "没有提升",
        "没有差别", "没有帮助", "不值", "放弃了", "停止", "走不通", "行不通"]
# 语气：口语/自我
COLLOQ = ["其实", "说实话", "老实说", "折腾", "搞", "弄", "挺", "蛮", "吧", "呢", "啊"]
HEDGE = ["可能", "或许", "大概", "似乎", "也许", "应该", "或许"]
# 句首连词
SENT_START_CONN = re.compile(r"(?:^|[。！？\n])\s*(?:但|而|因此|所以|于是|不过|然而|此外|另外|同时|而且|并且|总之|至于)")

NAMES = [
    "reveal", "causal", "contrast", "define", "eval", "colloquial", "hedge",
    "semicolon", "long_chain", "sent_start_conn", "conclusion_wrap",
    "avg_sent_len", "declarative_ratio", "i_ratio", "question_ratio",
]


def _rate(cnt, n):
    return 1000.0 * cnt / max(1, n)


def extract(text):
    t = text
    n = len(t)
    sents = [s for s in re.split(r"[。！？\n]+", t) if s.strip()]
    ns = max(1, len(sents))

    reveal = sum(t.count(p) for p in REVEAL)
    causal = sum(t.count(p) for p in CAUSAL)
    contrast = sum(len(re.findall(p, t)) for p in CONTRAST)
    define = sum(t.count(p) for p in DEFINE)
    ev = sum(t.count(p) for p in EVAL)
    colloq = sum(t.count(p) for p in COLLOQ)
    hedge = sum(t.count(p) for p in HEDGE)
    semicolon = t.count("；")
    # 最长并列链：以，；分隔的连续小句
    max_chain = 0
    for seg in re.split(r"[。！？\n]", t):
        chain = len(re.split(r"[，；、]", seg))
        max_chain = max(max_chain, chain)
    start_conn = len(SENT_START_CONN.findall(t))
    tail = t[int(n * 0.8):]
    wrap = sum(tail.count(p) for p in ["总结", "总的来说", "综上", "真正", "其余", "以上", "归根", "总的来说", "这轮", "最后"])
    slens = [len(s) for s in sents] or [0]
    question = len(re.findall(r"[？?]", t))
    return np.array([
        _rate(reveal, n), _rate(causal, n), _rate(contrast, n), _rate(define, n),
        _rate(ev, n), _rate(colloq, n), _rate(hedge, n), _rate(semicolon, n),
        max_chain, start_conn / ns, wrap,
        float(np.mean(slens)), 1.0,          # declarative_ratio 占位
        (t.count("我") - t.count("我们")) / max(1, n), question / ns,
    ], dtype=np.float32)


def extract_many(texts):
    return np.stack([extract(x) for x in texts])


if __name__ == "__main__":
    import sys, json
    from style_features import FEATURE_NAMES  # noqa
    f = extract(sys.stdin.read())
    for k, v in zip(NAMES, f):
        print(f"{k:18s} {v:.4f}")
