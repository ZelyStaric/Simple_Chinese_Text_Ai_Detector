#!/usr/bin/env python3
"""抓 840f.com 上转载的真人「职场/成长/文案」类文章，作为公众号风人类语料。"""
import os, re, json, time, urllib.request, urllib.parse
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "human", "human_wechat.jsonl")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
CJK = re.compile(r"[\u4e00-\u9fff]")
BASE = "https://www.840f.com"


def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read().decode("utf-8", "ignore")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def cjk_ratio(s):
    return len(CJK.findall(s)) / max(1, len(s))


def body(html):
    s = BeautifulSoup(html, "lxml")
    for b in s(["script", "style", "nav", "header", "footer", "aside", "form"]):
        b.decompose()
    best, bl = None, 0
    for sel in ["article", "div.article-content", "div.content", "div.post", "div#content", "main", "div"]:
        for el in s.select(sel):
            t = "\n".join(p.get_text(" ", strip=True) for p in el.find_all("p"))
            if len(t) > bl:
                best, bl = t, len(t)
    if best is None or bl < 300:
        best = "\n".join(p.get_text(" ", strip=True) for p in s.find_all("p"))
    title = (s.find("h1").get_text(strip=True) if s.find("h1") else "")
    return title, re.sub(r"\n{2,}", "\n", best).strip()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # 收集文章链接
    cats = ["zhichangjinjie", "daimarensheng", "wenanxiezuo", "yunyingsiwei", "zhiboyunying", "xiezuo", "chuangye"]
    urls = []
    # 主索引页含大量跨分类文章链接
    try:
        for a in BeautifulSoup(get(f"{BASE}/news/"), "lxml").find_all("a", href=True):
            if re.match(r"/news/[a-z]+/\d+$", a["href"]):
                urls.append(urllib.parse.urljoin(BASE, a["href"]))
    except Exception:
        pass
    for c in cats:
        for pg in range(1, 6):
            u = f"{BASE}/news/{c}/" if pg == 1 else f"{BASE}/news/{c}/page/{pg}.html"
            try:
                h = get(u)
            except Exception:
                continue
            for a in BeautifulSoup(h, "lxml").find_all("a", href=True):
                m = re.match(rf"/news/{c}/(\d+)$", a["href"])
                if m:
                    urls.append(urllib.parse.urljoin(BASE, a["href"]))
            time.sleep(0.2)
    urls = list(dict.fromkeys(urls))
    print("collected", len(urls))
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for u in urls:
            try:
                title, t = body(get(u))
            except Exception:
                continue
            if not (300 <= len(t) <= 6000) or cjk_ratio(t) < 0.5:
                continue
            f.write(json.dumps(dict(text=t, title=title, label=0, generator="human",
                                    domain="wechat", source="840f"), ensure_ascii=False) + "\n")
            f.flush(); n += 1
            time.sleep(0.3)
    print("wrote", n, "->", OUT)


if __name__ == "__main__":
    main()
