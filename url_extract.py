from __future__ import annotations

import re
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup


def fetch_html(url: str, timeout: int = 20) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    headers = {"User-Agent": "Mozilla/5.0 (BiasAnalyzer/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if not r.ok:
            return None, r.status_code, f"HTTP {r.status_code}"
        return r.text, r.status_code, None
    except Exception as e:
        return None, None, str(e)


def _clean(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_title_and_text(html: str) -> Tuple[Optional[str], str]:
    soup = BeautifulSoup(html, "html.parser")

    # drop noisy tags
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = None
    if soup.title and soup.title.string:
        title = _clean(soup.title.string)

    # best-effort: prefer <article>
    node = soup.find("article") or soup.body or soup

    # paragraphs usually hold main content
    paras = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    text = _clean(" ".join(paras))

    # fallback: all visible text (shortened)
    if len(text) < 300:
        text = _clean(node.get_text(" ", strip=True))

    return title, text