#!/usr/bin/env python3
"""抓取「确定是真人写」的中文技术博客文章，作为人类锚点。

来源：阮一峰、云风、酷壳(陈皓)、美团技术团队、廖雪峰、draven、张鑫旭。
优先 RSS + sitemap 扩展；只留中文长文；可选按日期过滤避开 AI 污染期。
"""
import os, re, json, time, gzip, argparse, urllib.request, urllib.parse
from datetime import datetime
from collections import Counter
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "human", "tech_blog.jsonl")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
CJK = re.compile(r"[\u4e00-\u9fff]")

SOURCES = {
    "ruanyifeng": ("https://www.ruanyifeng.com/blog/atom.xml",
                   "https://www.ruanyifeng.com/blog/", None),
    "yf": ("https://blog.codingnow.com/atom.xml", "https://blog.codingnow.com", None),
    "coolshell": ("https://coolshell.cn/feed", "https://coolshell.cn", "2023-06-01"),
    "meituan": ("https://tech.meituan.com/feed/", "https://tech.meituan.com", "2023-06-01"),
    "liaoxuefeng": ("https://www.liaoxuefeng.com/feed", "https://www.liaoxuefeng.com", "2023-06-01"),
    "draven": ("https://draven.co/feed.xml", "https://draven.co", None),
    "zhangxinxu": ("https://www.zhangxinxu.com/wordpress/feed/", "https://www.zhangxinxu.com", None),
}


def get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", "ignore")


def parse_feed(xml, base):
    links = []
    try:
        soup = BeautifulSoup(xml, "xml")
    except Exception:
        soup = BeautifulSoup(xml, "html.parser")
    for item in soup.find_all(["item", "entry"]):
        a = item.find("link")
        if a is None:
            continue
        href = a.get("href") or a.text.strip()
        if not href:
            continue
        links.append(urllib.parse.urljoin(base, href.strip()))
        if len(links) >= 200:
            break
    return links


def sitemap_links(base, limit=400):
    out = []
    for sm in ["/sitemap.xml", "/sitemap_index.xml"]:
        try:
            x = get(base + sm)
        except Exception:
            continue
        for m in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", x):
            if m.endswith(".xml"):
                try:
                    x2 = get(m)
                    out += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", x2)
                except Exception:
                    pass
            else:
                out.append(m)
        if out:
            break
    return out[:limit]


def extract_text(html):
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        bad.decompose()
    best, best_len = None, 0
    for sel in ["article", "main", "div.post-content", "div.entry-content", "div.md-content",
                "div[class*=content]", "div[class*=post]", "section"]:
        for el in soup.select(sel):
            txt = "\n".join(p.get_text(" ", strip=True) for p in el.find_all(["p", "li", "h2", "h3", "pre"]))
            if len(txt) > best_len:
                best, best_len = txt, len(txt)
    if best is None or best_len < 300:
        best = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    # 第一段标题
    h = soup.find("h1")
    title = h.get_text(strip=True) if h else ""
    return title, re.sub(r"\n{3,}", "\n\n", best).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_source", type=int, default=120)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "w", encoding="utf-8")
    seen = set()
    total = 0
    for name in a.sources:
        feed, base, cutoff = SOURCES[name]
        try:
            links = parse_feed(get(feed), base)
        except Exception as e:
            print(f"{name}: feed 失败 {e}")
            links = []
        if len(links) < 30:
            links += sitemap_links(base)
        links = list(dict.fromkeys(links))[:a.per_source]
        ok = 0
        for u in links:
            if u in seen:
                continue
            seen.add(u)
            try:
                html = get(u)
            except Exception:
                continue
            title, text = extract_text(html)
            if not text or not (300 <= len(text) <= 12000):
                continue
            if len(CJK.findall(text)) / max(1, len(text)) < 0.30:
                continue
            if text.count("{") + text.count("</") > 20:   # 代码/模板噪声过多
                continue
            f.write(json.dumps(dict(text=text, title=title, url=u, label=0,
                                    generator="human", domain="tech_blog",
                                    source=name), ensure_ascii=False) + "\n")
            f.flush()
            ok += 1; total += 1
            time.sleep(0.4)
        print(f"{name:14s} got {ok}")
    f.close()
    print(f"total {total} -> {OUT}")


if __name__ == "__main__":
    main()
