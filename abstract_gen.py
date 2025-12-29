#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import requests
from typing import List, Dict, Tuple

# ================= CONFIG =================
MODEL = "gpt-4o-mini"
ENDPOINT = "https://api.openai.com/v1/responses"

INPUT_CSV = "arxiv_strict_ml_100.csv"
OUTPUT_CSV = "arxiv_strict_ml_ai_abstracts.csv"

MIN_WC = 150
MAX_WC = 200
TARGET_WC = 170

# ================= HELPERS =================
_word_re = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

def word_count(text: str) -> int:
    return len(_word_re.findall(text))

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def ensure_sentence_end(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text[-1] in ".!?" else (text + ".")

def trim_to_words(text: str, n: int) -> str:
    """Deterministically trim to ~n words, keeping punctuation reasonably intact."""
    text = text.strip()
    words = _word_re.findall(text)
    if len(words) <= n:
        return normalize_ws(ensure_sentence_end(text))

    count = 0
    out_chars: List[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|\s+|[^\w\s]", text):
        token = m.group(0)
        if _word_re.fullmatch(token):
            count += 1
            if count > n:
                break
        out_chars.append(token)

    trimmed = "".join(out_chars).strip()
    return normalize_ws(ensure_sentence_end(trimmed))

def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "title" not in reader.fieldnames:
            raise RuntimeError(f"CSV must contain a 'title' column. Found: {reader.fieldnames}")
        return [row for row in reader]

def write_output(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "ai generated abstract"])
        writer.writeheader()
        writer.writerows(rows)

# ================= OPENAI CALL =================
def call_openai(payload: Dict, api_key: str) -> Dict:
    backoff = 2
    for _ in range(10):
        r = requests.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=60,
        )

        if r.status_code == 429:
            print(f"[429] Rate limited. Waiting {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError("Too many rate-limit retries.")

def extract_text(resp: Dict) -> str:
    parts = []
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if "text" in c and isinstance(c["text"], str):
                parts.append(c["text"])
    if not parts:
        raise RuntimeError("No text returned from model.")
    return "".join(parts)

# ================= GENERATION =================
def generate_two_abstracts(title: str, api_key: str) -> Tuple[str, str]:
    instructions = (
        "Generate two abstracts for the given title.\n"
        "Do not try to imitate the human writing style.\n"
        "Each abstract must be written as one paragraph only.\n"
        "Each abstract must be between 150 and 200 words.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "abstract_1": {"type": "string"},
            "abstract_2": {"type": "string"},
        },
        "required": ["abstract_1", "abstract_2"],
        "additionalProperties": False,
    }

    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Title: {title}"},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "abstract_pair",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 900,
    }

    resp = call_openai(payload, api_key)
    data = json.loads(extract_text(resp))

    a1 = normalize_ws(data["abstract_1"])
    a2 = normalize_ws(data["abstract_2"])

    # Enforce max length locally (no extra constraints added to prompt)
    if word_count(a1) > MAX_WC:
        a1 = trim_to_words(a1, TARGET_WC)
    else:
        a1 = normalize_ws(ensure_sentence_end(a1))

    if word_count(a2) > MAX_WC:
        a2 = trim_to_words(a2, TARGET_WC)
    else:
        a2 = normalize_ws(ensure_sentence_end(a2))

    return a1, a2

def combine_same_cell(a1: str, a2: str) -> str:
    """
    Faculty format: abstract1 ends, immediately continue abstract2 in same paragraph.
    i.e., one space between them (not a newline, not labels).
    """
    a1 = normalize_ws(ensure_sentence_end(a1))
    a2 = normalize_ws(a2)
    return f"{a1} {a2}"

# ================= MAIN =================
def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    input_rows = read_rows(INPUT_CSV)
    total = len(input_rows)
    if total == 0:
        print("No rows found in input CSV.", file=sys.stderr)
        return 1

    out_rows: List[Dict[str, str]] = []

    for i, row in enumerate(input_rows, 1):
        title = (row.get("title") or "").strip()
        if not title:
            out_rows.append({"title": "", "ai generated abstract": "ERROR: missing title"})
            print(f"[{i}/{total}] SKIP missing title", file=sys.stderr)
            continue

        try:
            a1, a2 = generate_two_abstracts(title, api_key)
            combined = combine_same_cell(a1, a2)
            out_rows.append({"title": title, "ai generated abstract": combined})
            print(
                f"[{i}/{total}] OK  WC1={word_count(a1)} WC2={word_count(a2)}",
                file=sys.stderr,
            )
        except Exception as e:
            out_rows.append({"title": title, "ai generated abstract": f"ERROR: {e}"})
            print(f"[{i}/{total}] FAIL {e}", file=sys.stderr)

        # small pacing; not tied to "50" or any hardcoded count
        time.sleep(0.1)

    write_output(OUTPUT_CSV, out_rows)
    print(f"Saved: {OUTPUT_CSV}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
