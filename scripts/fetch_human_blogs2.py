#!/usr/bin/env python3
"""抓真人中文技术博客（v2）：走年月归档/分页，默认只收 <=2022 年（避开 AI 污染）。

来源：阮一峰、云风、张鑫旭、美团技术团队、廖雪峰、Draven、酷壳(若可达)。
"""
import os, re, json, time, argparse, urllib.request, urllib.parse
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "human", "tech_blog.jsonl")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
CJK = re.compile(r"[\u4e00-\u9fff]")
YEAR_MAX = 2022


def months(base, y0, y1):
    return [f"{base}/{y}/{m:02d}/" for y in range(y0, y1 + 1) for m in range(1, 13)]


def pages(base, n):
    return [f"{base}/page/{i}/" for i in range(1, n + 1)]


CONF = {
    "ruanyifeng": dict(seeds=months("https://www.ruanyifeng.com/blog", 2005, YEAR_MAX),
                       art=r"/blog/\d{4}/\d{2}/[^/]+\.html$", host="www.ruanyifeng.com", cap=250),
    "yf": dict(seeds=months("https://blog.codingnow.com", 2005, YEAR_MAX),
               art=r"/\d{4}/\d{2}/[^/]+\.html$", host="blog.codingnow.com", cap=250),
    "zhangxinxu": dict(seeds=pages("https://www.zhangxinxu.com/wordpress", 45),
                       art=r"/wordpress/\d{4}/\d{2}/", host="www.zhangxinxu.com", cap=200),
    "meituan": dict(seeds=pages("https://tech.meituan.com", 45),
                    art=r"/20\d\d/\d{2}/\d{2}/[^/]+\.html$", host="tech.meituan.com", cap=200),
    "liaoxuefeng": dict(seeds=["https://www.liaoxuefeng.com/wiki/"],
                        art=r"/wiki/[0-9a-f]{20,}/[^/]+$", host="www.liaoxuefeng.com", cap=150),
    "draven": dict(seeds=[], sitemap="https://draven.co/sitemap.xml",
                   art=r"draven\.co/[^/]+/?$", host="draven.co", cap=150),
}


def get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def year_ok(u):
    m = re.search(r"/(20\d\d)/", u)
    return (not m) or (int(m.group(1)) <= YEAR_MAX)


def discover(name, c):
    found, seen = [], set()
    if c.get("sitemap"):
        x = get(c["sitemap"])
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", x)
        found = [u for u in urls if re.search(c["art"], u)]
        return list(dict.fromkeys(found))[:c["cap"]]
    for s in c["seeds"]:
        if len(found) >= c["cap"]:
            break
        try:
            h = get(s)
        except Exception:
            continue
        for a in BeautifulSoup(h, "lxml").find_all("a", href=True):
            u = urllib.parse.urljoin(s, a["href"]).split("#")[0]
            if urllib.parse.urlparse(u).netloc != c["host"]:
                continue
            if u in seen:
                continue
            if re.search(c["art"], u) and year_ok(u):
                seen.add(u); found.append(u)
                if len(found) >= c["cap"]:
                    break
    return found


def extract_text(html):
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        bad.decompose()
    best, bl = None, 0
    for sel in ["article", "main", "div.post-content", "div.entry-content", "div.md-content",
                "div[class*=content]", "div[class*=post]", "section", "div#content"]:
        for el in soup.select(sel):
            txt = "\n".join(p.get_text(" ", strip=True) for p in el.find_all(["p", "li", "pre"]))
            if len(txt) > bl:
                best, bl = txt, len(txt)
    if best is None or bl < 300:
        best = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    h = soup.find("h1")
    return (h.get_text(strip=True) if h else ""), re.sub(r"\n{3,}", "\n\n", best).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=list(CONF))
    ap.add_argument("--cap", type=int, default=0, help="覆盖每源上限")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    existing = set()
    if os.path.exists(OUT):                      # 断点续抓：不覆盖、按 url 去重
        for line in open(OUT, encoding="utf-8"):
            try:
                existing.add(json.loads(line)["url"])
            except Exception:
                pass
        print(f"已存在 {len(existing)} 篇，追加模式")
    f = open(OUT, "a", encoding="utf-8")
    total = 0
    for name in a.sources:
        c = dict(CONF[name])
        if a.cap:
            c["cap"] = a.cap
        urls = discover(name, c)
        ok = 0
        for u in urls:
            if u in existing:
                continue
            try:
                html = get(u)
            except Exception:
                continue
            title, text = extract_text(html)
            if not text or not (300 <= len(text) <= 15000):
                continue
            if len(CJK.findall(text)) / max(1, len(text)) < 0.30:
                continue
            if text.count("{") + text.count("</") > 25:
                continue
            f.write(json.dumps(dict(text=text, title=title, url=u, label=0,
                                    generator="human", domain="tech_blog",
                                    source=name), ensure_ascii=False) + "\n")
            f.flush(); ok += 1; total += 1
            time.sleep(0.3)
        print(f"{name:14s} discovered {len(urls):4d} got {ok}", flush=True)
    f.close()
    print(f"total {total} -> {OUT}")


if __name__ == "__main__":
    main()
