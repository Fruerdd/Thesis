import re
from dataclasses import dataclass
from typing import Optional

# try to use requests if installed
try:
    import requests
except Exception:
    requests = None

# try to use bs4 if installed (best)
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


@dataclass
class ExtractResult:
    http_status: int | None
    fetch_error: str | None
    title: str | None
    raw_text: str | None
    used_text: str


def _html_to_text(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        # remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # fallback (rough)
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


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
            headers={
                "User-Agent": "Mozilla/5.0 (BiasDetectorBot/1.0)"
            },
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
        text = _html_to_text(html)

        title = None
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

        # used_text is what you feed into model
        used_text = text

        return ExtractResult(
            http_status=http_status,
            fetch_error=None,
            title=title,
            raw_text=text,
            used_text=used_text,
        )
    except Exception as e:
        return ExtractResult(
            http_status=None,
            fetch_error=str(e),
            title=None,
            raw_text=None,
            used_text="",
        )