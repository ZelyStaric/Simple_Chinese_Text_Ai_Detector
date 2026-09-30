#!/usr/bin/env python3
"""用你博客的同一个模型（opencode-go / deepseek-v4.1-flash）批量生成中文语料。

phase=article: 生成「技术复盘/踩坑/失败总结」类长文（label 1, AI）
phase=edit:    用同模型对上述文章去味改写（label 2, AI-edited）
"""
import os, sys, json, uuid, time, argparse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH = os.path.expanduser("~/.local/share/opencode/auth.json")
URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"

TOPICS = [
    "自建推理服务的踩坑复盘", "为什么我的优化没效果", "一次失败的技术选型", "本地跑大模型走不通的路",
    "性能调优的得与失", "被高估的方案", "被低估的简单方案", "重构不如重写的一次经历",
    "从零搭一套监控的代价", "追新框架的教训", "过度设计的代价", "内存泄漏排查复盘",
    "一次线上事故的复盘", "小团队做基础设施的取舍", "向量数据库选型的纠结", "KV 缓存优化的边界",
    "为什么我们没有上微服务", "分布式锁踩过的坑", "从单体到拆分的失败尝试", "构建速度优化的边际收益",
    "CI/CD 折腾一圈的结论", "自研 vs 用现成的权衡", "数据管道重构复盘", "前端构建工具迁移的代价",
    "一次数据库迁移的教训", "缓存策略反复调整的复盘", "限流方案的取舍", "日志系统的成本反思",
    "为什么放弃了这个架构", "可观测性投入产出比", "模型量化对质量的影响复盘", "GPU 利用率优化到什么程度就够了",
    "写测试的性价比", "代码评审的边界", "一次失败的重构", "过早优化的代价",
    "工具链整合的坑", "一次 API 设计返工", "状态管理的取舍", "并发模型的踩坑",
    "我把简单问题做复杂了", "这次我学到的三个教训", "复盘：一个没有按期完成的项目",
    "为什么这个功能最后砍掉了", "技术债偿还的真实成本", "从痛点出发反而走弯路",
    "一次扩容的失败经历", "把性能优化做到头之后的反思", "自建评测集的意义与代价", "复现别人的方案为什么失败",
]
WRAP_HEAD = ("请写一篇中文技术复盘/踩坑记录，主题「{t}」。要求：先给结论，再分析原因，"
             "结尾总结取舍并给出评价（可以说「不划算」「到顶了」「差距过大」这类判断）。"
             "700 到 1200 字，自然行文。")
EDIT_HEAD = ("请把下面这篇文章改写得更像真人写的、降低 AI 味：少用套话与排比、少用列表、"
             "少用总结式收尾，多写具体事实和第一人称判断，保持原意与技术细节。\n\n原文：\n")

WECHAT_TOPICS = [
    "AI 到底会取代哪些工作", "普通人如何抓住这波风口", "为什么越努力越焦虑", "这届年轻人开始整顿职场",
    "30 岁前必须明白的 5 件事", "我用这个方法一年读完了 50 本书", "副业刚需时代的生存指南",
    "别用战术上的勤奋掩盖战略上的懒惰", "高能量人士都在坚持的几个习惯", "断舍离之后我的生活变了",
    "为什么你总是很累", "真正的自律从来不靠意志力", "学会这几点你的表达会更有影响力",
    "大厂裁员之后我去了哪里", "月薪五千如何理财", "这个习惯正在悄悄毁掉你",
    "从负债到存款我做对了什么", "情绪稳定是成年人最顶级的能力", "赚钱的本质是什么",
    "为什么说认知决定命运", "把时间花在这三件事上", "一个普通人逆袭的真实路径",
    "信息差正在被抹平普通人怎么办", "为什么我劝你别急着买房", "如何用最小成本做一个副业",
]
WECHAT_HEAD = ("请写一篇微信公众号推文，主题「{t}」。要求：标题有吸引力，开头用一句有共鸣的话引入，"
               "分小段展开（每段 2-4 句），中间可以有要点式小结，结尾引导点赞在看转发并给出行动建议。"
               "语气亲近、有节奏感，500 到 900 字。")


def key():
    return json.load(open(AUTH))["opencode-go"]["key"]


def call(prompt, session, max_tokens=4500, retry=2):
    k = key()
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + k,
         "x-api-key": k, "x-opencode-session": session, "User-Agent": "opencode"}
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for _ in range(retry + 1):
        try:
            d = json.loads(urllib.request.urlopen(
                urllib.request.Request(URL, data=body, headers=h), timeout=300).read())
            m = d["choices"][0]["message"]
            c = (m.get("content") or "").strip()
            if len(c) >= 120:
                return c
            max_tokens += 3000
            body = json.dumps({"model": MODEL, "max_tokens": max_tokens,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
        except urllib.error.HTTPError as e:
            time.sleep(5)
        except Exception:
            time.sleep(5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["article", "edit"], required=True)
    ap.add_argument("--genre", choices=["tech", "wechat"], default="tech")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "generated", "self_article.jsonl"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "data", "generated",
                                "self_article.jsonl" if a.phase == "article" else "self_edited.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    session = str(uuid.uuid4())

    if a.phase == "article":
        if a.genre == "wechat":
            topics, head, dom = WECHAT_TOPICS, WECHAT_HEAD, "wechat"
        else:
            topics, head, dom = TOPICS, WRAP_HEAD, "tech_blog"
        items = [(topics[i % len(topics)] + (f"（第 {i//len(topics)+1} 篇）" if i >= len(topics) else ""),)
                 for i in range(a.n)]
        gen, lab = "deepseek-v4.1-flash", 1
        def work(it):
            t = it[0]
            c = call(head.format(t=t), session)
            return dict(text=c, label=lab, generator=gen, domain=dom,
                        source="self-gen", prompt=t) if c else None
    else:
        src = [json.loads(l) for l in open(a.src, encoding="utf-8")][:a.n]
        items = src
        gen, lab = "deepseek-v4.1-flash-edited", 2
        def work(it):
            c = call(EDIT_HEAD + it["text"], session)
            return dict(text=c, label=lab, generator=gen, domain="tech_blog",
                        source="self-gen", prompt="edit") if c else None

    f = open(out, "w", encoding="utf-8")
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if done % 20 == 0:
                print(f"[{done}/{len(items)}]", flush=True)
    f.close()
    print("wrote", sum(1 for _ in open(out)), "->", out)


if __name__ == "__main__":
    main()
