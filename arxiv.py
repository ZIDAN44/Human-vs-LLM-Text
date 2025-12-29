import csv
import html
import re
import time
from typing import Dict, List, Optional, Set

import feedparser
import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"

PRIMARY_ML: Set[str] = {"cs.LG", "stat.ML"}

EXCLUDE_CATS: Set[str] = {
    "cs.CV",  # Computer Vision
    "cs.CL",  # NLP / Computation and Language
    "cs.AI",  # AI (broad)
    "cs.RO",  # Robotics
    "cs.SI",  # Social and Information Networks
    "q-bio",  # Biology umbrella (prefix match handled)
    "q-fin",  # Finance umbrella
    "econ",   # Economics umbrella
    "eess",   # Electrical Engineering umbrella
    "physics" # Physics umbrella
}

REQUIRE_PHRASES: List[str] = [
    "machine learning",
    "deep learning",
    "neural network",
    "representation learning",
    "supervised learning",
    "unsupervised learning",
    "reinforcement learning",
    "bayesian",
    "gradient",
]

EXCLUDE_PHRASES: List[str] = [
    "survey",
    "review",
    "tutorial",
    "a primer",
    "systematic review",
    "meta-analysis",
]

# ---------- LaTeX → readable (non-destructive) ----------

LATEX_SYMBOLS: Dict[str, str] = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\theta": "theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\phi": "phi",
    r"\psi": "psi",
    r"\omega": "omega",
    r"\tau": "tau",
    r"\perp": "⊥",
    r"\times": "x",
    r"\cdot": "·",
}

# Precompiled regexes
CMD_WITH_ARG = re.compile(r"\\[a-zA-Z]+\*?\{([^}]*)\}")  # \emph{X} -> X
INLINE_CMD = re.compile(r"\\[a-zA-Z]+\*?")               # \alpha
MATH_DOLLAR = re.compile(r"\$([^$]+)\$")                 # $...$ -> ...
MATH_PAREN = re.compile(r"\\\((.*?)\\\)")                # \( ... \)
MATH_BRACK = re.compile(r"\\\[(.*?)\\\]")                # \[ ... \]
SUBSCRIPT = re.compile(r"_\{([^}]*)\}")                  # _{i} -> _i
SUPERSCRIPT = re.compile(r"\^\{([^}]*)\}")               # ^{2} -> ^2
SPACES = re.compile(r"\s+")

LATEX_OPEN_QUOTE = re.compile(r"``")
LATEX_CLOSE_QUOTE = re.compile(r"''")
STRAY_COMBINING_GRAVE_BEFORE_QUOTE = re.compile(r"(?:(?<=\s)|^)\u0300(?=[\"'])")


def clean_arxiv_text_readable(text: str) -> str:
    r"""
    Make arXiv LaTeX readable WITHOUT altering meaning (formatting-only).
    - Keeps math content; removes only delimiters ($..$, \(..\), \[..\])
    - Unwraps formatting macros like \emph{...}
    - Normalizes LaTeX quotes ``...'' → “...”
    """
    if not text:
        return ""

    text = html.unescape(text).replace("\r", " ").replace("\n", " ")

    # Keep math content, remove delimiters only
    text = MATH_DOLLAR.sub(r"\1", text)
    text = MATH_PAREN.sub(r"\1", text)
    text = MATH_BRACK.sub(r"\1", text)

    # Unwrap commands like \emph{...}, \text{...} (repeat for nesting)
    for _ in range(3):
        new = CMD_WITH_ARG.sub(r"\1", text)
        if new == text:
            break
        text = new

    # Preserve subscripts/superscripts (don’t let regex caret anchor corrupt text)
    text = SUBSCRIPT.sub(r"_\1", text)
    text = SUPERSCRIPT.sub(lambda m: "^" + m.group(1), text)

    # Replace known LaTeX symbols
    for k, v in LATEX_SYMBOLS.items():
        text = text.replace(k, v)

    # Remove remaining formatting commands only
    text = INLINE_CMD.sub("", text)

    # Remove leftover braces/backslashes
    text = text.replace("{", "").replace("}", "").replace("\\", "")

    # Normalize LaTeX quotes and a rare stray combining accent before quotes
    text = LATEX_OPEN_QUOTE.sub("“", text)
    text = LATEX_CLOSE_QUOTE.sub("”", text)
    text = STRAY_COMBINING_GRAVE_BEFORE_QUOTE.sub("", text)

    return SPACES.sub(" ", text).strip()


# ---------- arXiv parsing helpers ----------

def get_primary_category(entry) -> Optional[str]:
    pc = entry.get("arxiv_primary_category")
    return pc.get("term") if isinstance(pc, dict) else None


def get_all_categories(entry) -> List[str]:
    tags = entry.get("tags", [])
    return [t.get("term") for t in tags if isinstance(t, dict) and t.get("term")]


def looks_strict_ml(title: str, abstract: str) -> bool:
    text = (title + " " + abstract).lower()
    if not any(p in text for p in REQUIRE_PHRASES):
        return False
    if any(p in text for p in EXCLUDE_PHRASES):
        return False
    return True


def is_excluded_category(all_cats: List[str]) -> bool:
    for c in all_cats:
        if c in EXCLUDE_CATS:
            return True
        if c.split(".")[0] in EXCLUDE_CATS:
            return True
    return False


def get_pdf_link(entry) -> Optional[str]:
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href")
    return None


# ---------- main fetch + save ----------

def fetch_strict_ml_only(
    target_n: int = 100,
    page_size: int = 200,
    sleep_seconds: float = 3.0,
    user_agent_email: str = "your_email@example.com",
) -> List[Dict[str, str]]:
    collected: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()
    start = 0

    headers = {"User-Agent": f"strict-ml-only/1.0 (mailto:{user_agent_email})"}
    search_query = "(cat:cs.LG OR cat:stat.ML)"

    while len(collected) < target_n:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = requests.get(ARXIV_API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            arxiv_id = entry.id
            if arxiv_id in seen_ids:
                continue

            primary = get_primary_category(entry)
            if primary not in PRIMARY_ML:
                continue

            all_cats = get_all_categories(entry)
            if is_excluded_category(all_cats):
                continue

            raw_title = " ".join(entry.title.split())
            raw_abstract = " ".join(entry.summary.split())

            title = clean_arxiv_text_readable(raw_title)
            abstract = clean_arxiv_text_readable(raw_abstract)

            if not looks_strict_ml(title, abstract):
                continue

            pdf_link = get_pdf_link(entry)

            collected.append(
                {
                    "id": arxiv_id,
                    "link": pdf_link or "",
                    "title": title,
                    "human_written_abstract": abstract,
                }
            )
            seen_ids.add(arxiv_id)

            if len(collected) >= target_n:
                break

        start += page_size
        time.sleep(sleep_seconds)

    return collected


def save_csv(rows: List[Dict[str, str]], filename: str = "arxiv_strict_ml_100.csv") -> None:
    fieldnames = ["id", "link", "title", "human_written_abstract"]
    # utf-8-sig makes Excel open Unicode correctly
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    papers = fetch_strict_ml_only(target_n=100)
    save_csv(papers, "arxiv_strict_ml_100.csv")
    print(f"Saved {len(papers)} papers -> arxiv_strict_ml_100.csv")
