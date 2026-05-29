import re
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import trafilatura
    from trafilatura.settings import use_config
    _TRAF_CFG = use_config()
    _TRAF_CFG.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
except Exception:
    trafilatura = None
    _TRAF_CFG = None


# Real browser UA. Many news sites return bot-detection / paywall stubs
# to the previous "BiasDetectorBot" agent, which gave neutral lede-only text
# and caused the model to default to Center / Neutral.
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_MIN_BODY_CHARS = 400


@dataclass
class ExtractResult:
    http_status: int | None
    fetch_error: str | None
    title: str | None
    raw_text: str | None
    used_text: str


def _html_to_text_fallback(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "button"]
        ):
            tag.decompose()

        article = soup.find("article")
        if article is not None:
            text = article.get_text("\n")
        else:
            main = soup.find("main")
            if main is not None:
                text = main.get_text("\n")
            else:
                paragraphs = soup.find_all("p")
                if paragraphs:
                    text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)
                else:
                    text = soup.get_text("\n")

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_title(html: str) -> Optional[str]:
    if BeautifulSoup is None:
        m = re.search(r"<title>(.*?)</title>", html, flags=re.S | re.I)
        return m.group(1).strip() if m else None

    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()

    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        return tw["content"].strip()

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


def _extract_with_trafilatura(html: str, url: Optional[str]) -> Optional[str]:
    if trafilatura is None:
        return None
    try:
        text = trafilatura.extract(
            html,
            url=url,
            favor_recall=True,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
            no_fallback=False,
            config=_TRAF_CFG,
        )
    except Exception:
        return None
    if not text:
        return None
    text = text.strip()
    return text or None


def extract_from_url(url: str, timeout_sec: int = 12) -> ExtractResult:
    if requests is None:
        return ExtractResult(
            http_status=None,
            fetch_error="requests is not installed (pip install requests)",
            title=None,
            raw_text=None,
            used_text="",
        )

    try:
        resp = requests.get(
            url,
            timeout=timeout_sec,
            headers=_DEFAULT_HEADERS,
            allow_redirects=True,
        )
        http_status = resp.status_code
        if resp.status_code >= 400:
            return ExtractResult(
                http_status=http_status,
                fetch_error=f"HTTP {resp.status_code}",
                title=None,
                raw_text=None,
                used_text="",
            )

        html = resp.text
        title = _extract_title(html)

        body = _extract_with_trafilatura(html, url)
        used_source = "trafilatura"

        if not body or len(body) < _MIN_BODY_CHARS:
            fallback = _html_to_text_fallback(html)
            if fallback and (not body or len(fallback) > len(body)):
                body = fallback
                used_source = "fallback"

        body = (body or "").strip()

        if len(body) < _MIN_BODY_CHARS:
            return ExtractResult(
                http_status=http_status,
                fetch_error=(
                    f"Extracted body too short ({len(body)} chars via {used_source}); "
                    "likely paywall, JS-rendered content, or blocked by site."
                ),
                title=title,
                raw_text=body or None,
                used_text="",
            )

        return ExtractResult(
            http_status=http_status,
            fetch_error=None,
            title=title,
            raw_text=body,
            used_text=body,
        )
    except Exception as e:
        return ExtractResult(
            http_status=None,
            fetch_error=str(e),
            title=None,
            raw_text=None,
            used_text="",
        )
