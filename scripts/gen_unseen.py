#!/usr/bin/env python3
"""用本地 OpenAI 兼容端点（默认 KVMem Qwen3.8-27B @18200）生成 AI 文本，
作为「训练时未见过的生成器」测试集。

用法:
  ./env/bin/python scripts/gen_unseen.py --n 600 --workers 8 --out data/generated/qwen_qa.jsonl --kind qa
"""
import os, sys, json, argparse, random, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = os.path.expanduser("~/.config/qwen-kvmem.keys")

BLOG_TOPICS = [
    "分布式系统的一致性模型", "数据库索引为什么快", "缓存穿透与雪崩", "Go 的调度器", "Rust 所有权与借用",
    "Kubernetes 的调度", "消息队列的可靠性投递", "微服务的边界划分", "可观测性三件套", "零拷贝与内存映射",
    "Linux 虚拟内存", "TCP 拥塞控制", "HTTP/2 与 HTTP/3", "编译原理里的类型推导", "垃圾回收算法",
    "性能剖析的方法论", "重构遗留代码", "单元测试与可测试性", "幂等与重试", "分布式锁的实现",
    "协程与异步 IO", "容器网络与 CNI", "eBPF 观测", "向量数据库", "RAG 的工程实践",
]

NEWS_TOPICS = ["本地部署大模型", "国产 GPU 的软件生态", "开源与闭源模型的取舍", "AI 生成内容泛滥",
               "个人博客的衰落", "量化模型对硬件的要求", "推理框架的性能优化", "边缘设备跑大模型"]
SOCIAL_TOPICS = ["今天的通勤", "换了个新键盘", "周末爬山", "楼下新开的面馆", "熬夜改 bug",
                 "显卡又涨价了", "一起拼单买显卡", "公司团建", "手机续航", "追的剧完结了"]


def load_key():
    if os.path.exists(KEYS):
        k = open(KEYS).read().strip()
        return k if k.lower().startswith("bearer ") else f"Bearer {k}"
    return None


def read_hc3_questions(n):
    qs = []
    base = os.path.join(ROOT, "data", "raw", "hc3")
    for dom in ["open_qa", "baike", "finance", "law", "medicine", "psychology", "nlpcc_dbqa"]:
        p = os.path.join(base, f"{dom}.jsonl")
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            qs.append((d["question"], f"hc3_{dom}"))
    random.seed(0)
    random.shuffle(qs)
    return qs[:n]


def read_titles(topics_file, limit=200):
    out = []
    if topics_file and os.path.exists(topics_file):
        for line in open(topics_file, encoding="utf-8"):
            try:
                t = json.loads(line).get("title", "").strip()
            except Exception:
                t = ""
            if t and 4 <= len(t) <= 40 and "BLOG" not in t and "博客" not in t:
                out.append(t)
    return out[:limit]


def make_prompts(kind, n, topics_file="", edit_style="medium"):
    if kind == "qa":
        return [(q, dom) for q, dom in read_hc3_questions(n)]
    if kind == "blog":
        titles = list(BLOG_TOPICS)
        if topics_file and os.path.exists(topics_file):
            for line in open(topics_file, encoding="utf-8"):
                try:
                    t = json.loads(line).get("title", "").strip()
                except Exception:
                    t = ""
                if t and "BLOG" not in t and "博客" not in t and 4 <= len(t) <= 40:
                    titles.append(t)
        random.seed(0); random.shuffle(titles)
        return [(f"写一篇中文技术博客，主题是「{t}」。要求 700 到 1200 字，分小节展开，结合具体例子。",
                 "tech_blog") for t in titles[:n]]
    if kind == "flavor":
        titles = list(BLOG_TOPICS) + list(NEWS_TOPICS) + list(SOCIAL_TOPICS) + read_titles(topics_file)
        random.seed(0); random.shuffle(titles)
        out = []
        for t in titles[:n]:
            p = (f"写一篇关于「{t}」的文章，约 400 字。写作要求：开头先抛出结论；"
                 "中间分析原因，多用「原因很简单」「原因也清楚」「由此可见」「换句话说」这类过渡；"
                 "结尾用「真正有效的是……其余……」「总的来说」收束；"
                 "整体带评价口吻，可用「不划算」「差距过大」「到顶了」「并不新奇」等判断。")
            out.append((p, "tech_blog"))
        return out
    if kind == "review":
        titles = list(BLOG_TOPICS) + read_titles(topics_file)
        random.seed(0); random.shuffle(titles)
        return [(f"写一篇关于「{t}」的技术复盘与经验总结：先给结论、分析原因、"
                 f"最后总结取舍与评价，语气自然。", "tech_blog") for t in titles[:n]]
    if kind == "edit":
        src = []
        if topics_file and os.path.exists(topics_file):
            src = [json.loads(l) for l in open(topics_file, encoding="utf-8")]
        random.seed(0); random.shuffle(src)
        styles = {
            "light": "请把下面这篇文章改写得更自然一些，稍微降低 AI 味，保持原意与技术细节。\n\n",
            "medium": ("请把下面这篇文章改写得更像人类写的，降低 AI 味：少用套话与排比、"
                       "少用 markdown 列表、少用评价式总结，多写具体事实，保持原意与技术细节。\n\n"),
            "heavy": ("请彻底重写下面这篇文章，让它读起来完全像真人写的：去掉所有套话、排比、"
                      "清单化和总结式收尾，换成自然的散文叙述，保留技术事实。\n\n"),
        }
        head = styles.get(edit_style, styles["medium"])
        out = []
        for r in src[:n]:
            out.append((head + f"原文：\n{r['text']}", r.get("domain", "tech_blog")))
        return out
    if kind == "article":
        titles = []
        if topics_file and os.path.exists(topics_file):
            for line in open(topics_file, encoding="utf-8"):
                t = json.loads(line).get("title", "").strip()
                if t:
                    titles.append(t)
        else:
            titles = NEWS_TOPICS * (n // len(NEWS_TOPICS) + 1)
        random.seed(0)
        random.shuffle(titles)
        out = []
        for t in titles[:n]:
            out.append((f"请围绕「{t}」写一篇 300 到 600 字的中文科普文章。", "wiki_article"))
        return out
    random.seed(0)
    topics = NEWS_TOPICS if kind == "news" else SOCIAL_TOPICS
    out = []
    while len(out) < n:
        t = random.choice(topics)
        if kind == "news":
            p = f"用中文写一段关于「{t}」的新闻报道，200 字左右。"
        else:
            p = f"用中文以普通网友的口吻发一条关于「{t}」的社交动态，100 字左右。"
        out.append((p, f"{kind}_gen"))
    return out


def call(url, model, key, prompt, temp, max_tokens, timeout=180):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        **({"Authorization": key} if key else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    txt = (m.get("content") or "").strip()
    return txt, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18200/v1/chat/completions")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--kind", choices=["qa", "news", "social", "article", "blog", "edit", "review", "flavor"], default="qa")
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--edit_style", choices=["light", "medium", "heavy"], default="medium")
    ap.add_argument("--topics", default="")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=320)
    ap.add_argument("--model_tag", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = load_key()
    prompts = make_prompts(a.kind, a.n, a.topics, a.edit_style)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w", encoding="utf-8")
    done = 0

    def work(item):
        q, dom = item
        try:
            txt, usage = call(a.url, a.model, key, q, a.temp, a.max_tokens)
            if len(txt) < 30:
                return None
            return dict(text=txt, label=a.label, generator=a.model_tag,
                        domain=dom, source="local-gen", prompt=q)
        except Exception as e:
            return {"_err": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in prompts]
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            if r is None:
                continue
            if "_err" in r:
                if done <= 5 or done % 50 == 0:
                    print(f"[{done}/{len(prompts)}] err: {r['_err']}", flush=True)
                continue
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            if done % 25 == 0:
                print(f"[{done}/{len(prompts)}]", flush=True)
    fout.close()
    n = sum(1 for _ in open(a.out))
    print(f"\nwrote {n} -> {a.out}")


if __name__ == "__main__":
    main()
